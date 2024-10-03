import warnings
from argparse import ArgumentParser

import torch
import torchvision.transforms.v2 as transforms
from monai.networks.nets import DynUNet
from torch.utils.data import DataLoader

from torch_utils import network_utils
from torch_utils.misc import import_module_from_path
from training.dataset import CamusSegmentationDataset
from training.epoch_cycles import evaluate_segmentation_cycle

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    args = parser.parse_args()
    return args


def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)

    device = args.gpu_id if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    test_transforms = transforms.Compose([transforms.Resize((args.image_size, args.image_size))])

    kernels, strides = network_utils.get_kernels_strides(args.image_size)

    test_dataset = CamusSegmentationDataset(args.data_dir, 'testing', data_type=args.data_type)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers)

    model = DynUNet(spatial_dims=2,
                    in_channels=args.in_channels,
                    kernel_size=kernels,
                    strides=strides,
                    upsample_kernel_size=strides[1:],
                    res_block=args.res_block,
                    dropout=args.dropout,
                    out_channels=args.num_classes).to(device)

    if args.compile:
        model = torch.compile(model)

    assert args.load_path is not None, 'Please provide a path to a pretrained model'
    state_dict = torch.load(args.load_path, weights_only=False)
    remove_prefix = '_orig_mod.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f'Model loaded from {args.load_path}')

    [test_mean_dice,
     test_mean_per_label_dice,
     test_std_per_label_dice,
     test_mean_hd,
     test_std_hd] = evaluate_segmentation_cycle(test_data_loader, test_transforms, device, model)

    print(args.load_path)
    print(
        f'Test Dice: [Lv endo {test_mean_per_label_dice[1] * 100:.2f}]'
        f' [Lv Epi {test_mean_per_label_dice[2] * 100:.2f}]'
        f' [Lv Atrium {test_mean_per_label_dice[0] * 100:.2f}]'
        f' [Mean: {test_mean_dice * 100:.2f}]'
        f' [Lv endo std {test_std_per_label_dice[1] * 100:.2f}]'
        f' [Lv Epi std {test_std_per_label_dice[2] * 100:.2f}]'
        f' [Lv Atrium std {test_std_per_label_dice[0] * 100:.2f}] '
        f' [Lv endo mean Hausdorff {test_mean_hd:.2f}]'
        f' [Lv endo std Hausdorff {test_std_hd:.2f}]'
    )


if __name__ == '__main__':
    input_args = get_args_from_command_line()
    main(input_args)
