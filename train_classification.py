import os
import time
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet_v2_s

from torch_utils import network_utils
from torch_utils.misc import import_module_from_path
from training.dataset import CamusClassificationDataset
from training.epoch_cycles import train_classifier_cycle, evaluate_classifier_cycle


def get_args_from_command_line():
    config_parser = ArgumentParser(description="Load config file", add_help=False)
    config_parser.add_argument('--config',
                               type=str,
                               help="Path to the config file")
    args, remaining_argv = config_parser.parse_known_args()
    if args.config:
        cfg = import_module_from_path('cfg', args.config).cfg
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
    parser.add_argument('--in-channels',
                        dest='in_channels',
                        help='Number of input image channels',
                        default=cfg.NETWORK.IN_CHANNELS,
                        type=int)
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
         transforms.RandAugment(fill=0, interpolation=transforms.InterpolationMode.BILINEAR)
         ])

    val_transforms = transforms.Compose([transforms.Resize((args.image_size, args.image_size))])

    model = efficientnet_v2_s()
    model.features[0][0] = nn.Conv2d(args.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                     bias=False)

    model.classifier[1] = torch.nn.Sequential(nn.Linear(in_features=1280, out_features=args.num_classes, bias=True))

    model = model.to(device)
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
        print(f'Model loaded from {args.load_path}')

    train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logs_path, 'val'))
    test_writer = SummaryWriter(os.path.join(logs_path, 'test'))

    train_dataset = CamusClassificationDataset(args.data_dir, 'training', data_type=args.data_type)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=args.shuffle,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers)

    val_dataset = CamusClassificationDataset(args.data_dir, 'validation', data_type=args.data_type)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 pin_memory=args.pin_memory,
                                 num_workers=args.num_workers)

    test_dataset = CamusClassificationDataset(args.data_dir, 'testing', data_type=args.data_type)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=False)

    best_acc = 0
    best_epoch = 0
    scaler = torch.amp.GradScaler()

    start_time = time.time()
    tracker = []
    for epoch in range(args.num_epochs + 1):
        train_mean_loss, train_report = train_classifier_cycle(model,
                                                               train_data_loader,
                                                               train_transforms,
                                                               criterion,
                                                               optimizer,
                                                               device,
                                                               scaler)
        val_report, val_auroc = evaluate_classifier_cycle(model,
                                                          val_data_loader,
                                                          val_transforms,
                                                          args.num_classes,
                                                          device)
        test_report, test_auroc = evaluate_classifier_cycle(model,
                                                            test_data_loader,
                                                            val_transforms,
                                                            args.num_classes,
                                                            device)

        tracker += [[epoch,
                     train_mean_loss,
                     train_report['accuracy'],
                     val_report['accuracy'],
                     test_report['accuracy']]]

        print(
            f'\nEpoch {epoch}:'
            f' [Train Loss: {train_mean_loss:.4f}]'
            f' [Train Acc: {train_report["accuracy"]:.4f}]'
            f' [Val Acc: {val_report["accuracy"]:.4f}]'
            f' [Test Acc: {test_report["accuracy"]:.4f}]')

        os.makedirs(args.results_dir, exist_ok=True)

        train_writer.add_scalar('Accuracy', train_report['accuracy'], epoch)
        val_writer.add_scalar('Accuracy', val_report['accuracy'], epoch)
        test_writer.add_scalar('Accuracy', test_report['accuracy'], epoch)

        if val_report['accuracy'] > best_acc:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.results_dir, 'model.pth'))
            best_acc = val_report['accuracy']

            print(network_utils.pretty_classification_report(test_report))
            print("===== saved best model ======")

    time_taken = time.time() - start_time
    time_taken = time.strftime("%H:%M:%S", time.gmtime(time_taken))
    print(f'Best epoch: {best_epoch}, best mean acc: {best_acc}, time taken: {time_taken}')

    training_df = pd.DataFrame(tracker, columns=['epoch', 'train_loss', 'train_acc', 'val_acc', 'test_acc'])
    training_df.to_excel(os.path.join(logs_path, 'train_log.xlsx'))

    train_writer.close()
    val_writer.close()
    test_writer.close()


if __name__ == '__main__':
    input_args = get_args_from_command_line()
    main(input_args)
