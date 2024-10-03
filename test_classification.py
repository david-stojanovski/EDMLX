import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm

from torch_utils.misc import import_module_from_path
from training.dataset import CamusClassificationDataset
from training.epoch_cycles import evaluate_classifier_cycle

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
    ############################################ Test Parameters ############################################
    parser.add_argument('--num-repeats',
                        dest='num_repeats',
                        default=cfg.TEST.NUM_REPEATS,
                        type=int)
    parser.add_argument('--subset-frac',
                        dest='subset_frac',
                        default=cfg.TEST.SUBSET_FRAC,
                        type=float)

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

    model = efficientnet_v2_s()
    model.features[0][0] = nn.Conv2d(args.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                     bias=False)
    model.classifier[1] = torch.nn.Sequential(nn.Linear(in_features=1280, out_features=args.num_classes, bias=True))

    model = model.to(device)
    if args.compile:
        try:
            print('Compiling model')
            model = torch.compile(model, dynamic=False)
            print('Successfully compiled model')
        except Exception as e:
            print('Failed model compilation, continuing without a compiled model')

    assert args.load_path is not None, 'Please provide a path to a pretrained model'
    state_dict = torch.load(args.load_path, weights_only=False)
    remove_prefix = '_orig_mod.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f'Model loaded from {args.load_path}')

    acc_tracker = []
    with tqdm(range(args.num_repeats), unit="repeat") as tepoch:
        for _ in tepoch:
            test_dataset = CamusClassificationDataset(args.data_dir, 'testing', data_type=args.data_type)
            test_dataset_subset = Subset(test_dataset,
                                         torch.randperm(len(test_dataset))[:int(len(test_dataset) * args.subset_frac)])
            test_data_loader = DataLoader(test_dataset_subset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=args.pin_memory,
                                          num_workers=args.num_workers)

            test_report, _ = evaluate_classifier_cycle(model,
                                                       test_data_loader,
                                                       test_transforms,
                                                       args.num_classes,
                                                       device)
            acc_tracker.append(test_report['accuracy'])

    bootstrap_mean = np.mean(acc_tracker, axis=0) * 100
    bootstrap_median = np.median(acc_tracker, axis=0) * 100
    bootstrap_std = np.std(acc_tracker, axis=0) * 100

    print(
        f'{args.num_repeats} repeats,'f' {args.subset_frac} fraction split: '
        f' [Mean acc: {bootstrap_mean:.2f}]'
        f' [Median acc: {bootstrap_median:.2f}]'
        f' [STD acc: {bootstrap_std:.2f}]')


if __name__ == '__main__':
    input_args = get_args_from_command_line()
    main(input_args)
