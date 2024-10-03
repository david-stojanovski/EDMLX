import os
import time
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from monai.networks.nets import DynUNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch_utils.misc
from torch_utils import network_utils
from training.dataset import CamusSegmentationDataset
from training.epoch_cycles import train_segmentation_cycle, evaluate_segmentation_cycle


def get_args_from_command_line():
    config_parser = ArgumentParser(description="Load config file", add_help=False)
    config_parser.add_argument('--config',
                               type=str,
                               help="Path to the config file")
    args, remaining_argv = config_parser.parse_known_args()
    if args.config:
        cfg = torch_utils.misc.import_module_from_path('cfg', args.config).cfg
    else:
        cfg = {}

    parser = ArgumentParser(
        parents=[config_parser],
        description="Script with configurable defaults"
    )

    ############################################ Dataset Parameters ############################################
    parser.add_argument('--data-dir',
                        dest='data_dir',
                        help='path to data directory',
                        default=cfg.DATASET.DATA_DIR,
                        type=str)
    parser.add_argument('--data-type',
                        dest='data_type',
                        help='image file format e.g. .png',
                        default=cfg.DATASET.DATA_TYPE,
                        type=str)
    parser.add_argument('--results-dir',
                        dest='results_dir',
                        help='path to save directory',
                        default=cfg.DATASET.RESULTS_DIR,
                        type=str)
    parser.add_argument('--image-size',
                        dest='image_size',
                        help='size of input images',
                        default=cfg.DATASET.IMAGE_SIZE,
                        type=int)
    ############################################ Network Parameters ############################################
    parser.add_argument('--num-classes',
                        dest='num_classes',
                        help='number of label classes',
                        default=cfg.NETWORK.NUM_CLASSES,
                        type=int)
    parser.add_argument('--dropout',
                        help='dropout probability of network',
                        default=cfg.NETWORK.DROPOUT,
                        type=float)
    parser.add_argument('--in-channels',
                        dest='in_channels',
                        help='Number of image input channels',
                        default=cfg.NETWORK.IN_CHANNELS,
                        type=int)
    parser.add_argument('--res-block',
                        dest='res_block',
                        help='Whether include residual blocks in network',
                        default=cfg.NETWORK.RES_BLOCK,
                        type=bool)
    parser.add_argument('--compile',
                        dest='compile',
                        help='Whether to perform pytorch 2.X automatic model compilation',
                        default=cfg.NETWORK.COMPILE,
                        type=bool)
    parser.add_argument('--load-path',
                        dest='load_path',
                        help='path to pretrained model',
                        default=cfg.NETWORK.LOAD_PATH)
    ############################################ Const Parameters ############################################
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--num-workers',
                        dest='num_workers',
                        help='number of cpu cores to use',
                        default=cfg.CONST.NUM_WORKERS,
                        type=int)
    parser.add_argument('--pin-memory',
                        dest='pin_memory',
                        help='Whether to pin memory of dataloaders',
                        default=cfg.CONST.PIN_MEMORY,
                        type=bool)
    parser.add_argument('--gpu-id',
                        dest='gpu_id',
                        help='GPU device id to use [cuda:0]',
                        default=cfg.CONST.GPU_ID,
                        type=str)
    ############################################ Train Parameters ############################################
    parser.add_argument('--num-epochs',
                        dest='num_epochs',
                        help='number of epochs to run training for',
                        default=cfg.TRAIN.NUM_EPOCHS,
                        type=int)
    parser.add_argument('--lr',
                        dest='lr',
                        help='learning rate',
                        default=cfg.TRAIN.LR,
                        type=float)
    parser.add_argument('--shuffle',
                        dest='shuffle',
                        help='Whether to shuffle training data',
                        default=cfg.TRAIN.SHUFFLE,
                        type=bool)
    parser.add_argument('--seed',
                        help='GPU device id to use [cuda0]',
                        default=cfg.TRAIN.SEED)

    args = parser.parse_args()
    return args


def main(args):
    logs_path = os.path.join(args.results_dir, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    network_utils.save_model_config(args, args.results_dir)

    if args.seed is not None:
        torch.manual_seed(int(args.seed))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)

    device = args.gpu_id if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_transforms = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size)),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),  # in degrees
         transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2),
                                 interpolation=transforms.InterpolationMode.BILINEAR),
         ])

    val_transforms = transforms.Compose([transforms.Resize((args.image_size, args.image_size))])

    kernels, strides = network_utils.get_kernels_strides(args.image_size)

    model = DynUNet(spatial_dims=2,
                    in_channels=args.in_channels,
                    kernel_size=kernels,
                    strides=strides,
                    upsample_kernel_size=strides[1:],
                    res_block=args.res_block,
                    dropout=args.dropout,
                    out_channels=args.num_classes).to(device)

    if args.compile:
        try:
            model = torch.compile(model, dynamic=False)
        except Exception as e:
            print('Failed to compile mode, continuing without compiled model')
    if args.load_path:
        state_dict = torch.load(args.load_path, weights_only=False)
        remove_prefix = '_orig_mod.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args.load_model_path}')

    logs_path = os.path.join(args.results_dir, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logs_path, 'val'))
    test_writer = SummaryWriter(os.path.join(logs_path, 'test'))

    train_dataset = CamusSegmentationDataset(args.data_dir, 'training', data_type=args.data_type)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=args.shuffle,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers)

    val_dataset = CamusSegmentationDataset(args.data_dir, 'validation', data_type=args.data_type)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 pin_memory=args.pin_memory,
                                 num_workers=args.num_workers)

    test_dataset = CamusSegmentationDataset(args.data_dir, 'testing', data_type=args.data_type)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    best_dice = 0
    best_epoch = 0
    scaler = torch.amp.GradScaler()

    start_time = time.time()
    tracker = []
    for epoch in range(args.num_epochs + 1):
        train_loss, train_dice = train_segmentation_cycle(model,
                                                          train_data_loader,
                                                          train_transforms,
                                                          args.num_classes,
                                                          criterion,
                                                          optimizer,
                                                          device,
                                                          scaler)
        val_mean_dice, _, _, _, _ = evaluate_segmentation_cycle(val_data_loader,
                                                                val_transforms,
                                                                device, model)
        test_mean_dice, test_mean_label_dice, _, _, _ = evaluate_segmentation_cycle(test_data_loader,
                                                                                    val_transforms,
                                                                                    device,
                                                                                    model)

        tracker += [[epoch,
                     train_loss,
                     train_dice,
                     val_mean_dice,
                     test_mean_dice]]

        print('\nEpoch %d: [Train Loss %.4f] [Train Dice: %.4f] [Val Dice: %.4f] [Test Dice %.4f]'
              % (epoch, train_loss, train_dice, val_mean_dice, test_mean_dice,))
        print('Epoch %d Test Dice: [Lv endo %.4f] [Lv Epi %.4f] [Lv Atrium %.4f] '
              % (epoch, test_mean_label_dice[1], test_mean_label_dice[2], test_mean_label_dice[0]))

        os.makedirs(args.results_dir, exist_ok=True)

        train_writer.add_scalar('Dice', train_dice, epoch)
        val_writer.add_scalar('Dice', val_mean_dice, epoch)
        test_writer.add_scalar('Dice', test_mean_dice, epoch)

        if val_mean_dice > best_dice:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.results_dir, 'model.pth'))
            best_dice = val_mean_dice
            print("===== saved best model ======")

    time_taken = time.time() - start_time
    time_taken = time.strftime("%H:%M:%S", time.gmtime(time_taken))
    print(f'Best epoch: {best_epoch}, best mean dice: {best_dice}, time taken: {time_taken}')

    training_df = pd.DataFrame(tracker, columns=['epoch', 'train_loss', 'train_dice', 'val_dice', 'test_dice'])
    training_df.to_excel(os.path.join(logs_path, 'train_log.xlsx'))

    train_writer.close()
    val_writer.close()
    test_writer.close()


if __name__ == '__main__':
    input_args = get_args_from_command_line()
    main(input_args)
