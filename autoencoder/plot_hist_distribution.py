from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import CamusVAEDataset


def train(args):
    val_dataset = CamusVAEDataset(args.data_dir, device='cpu', dataset_mode='train')
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize)

    hist_tracker = []
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, masks in tepoch:
            images = images / 255.0
            images = torch.nn.functional.interpolate(images, args.image_size)
            hist_tracker.append(images.numpy())

    hist_tracker = np.stack(hist_tracker).ravel()

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    plt.figure(figsize=(6.0, 6.0))
    plt.hist(hist_tracker,
             bins=255,
             range=(1 / 255., 1),
             # NOTE: This ignores zero values (so it doesn't have a big peak for the ultrasound sector)
             density=True,
             linewidth=1,
             edgecolor='black',
             histtype='stepfilled')
    plt.grid(False)
    plt.ylabel('Density of Probability', fontsize=18)
    plt.xlabel('Normalized Pixel Values', fontsize=18)
    plt.xticks(fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 3])
    plt.yticks(fontsize=16)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    plt.show()


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser for Echo downstream_tasks')
    parser.add_argument('--data_dir',
                        help='path to val folder',
                        type=str)
    parser.add_argument('--batchsize',
                        dest='batchsize',
                        help='batch size',
                        default=32,
                        type=int)
    parser.add_argument('--image-size',
                        dest='image_size',
                        help='Size of images for gathering histograms for',
                        default=256,
                        type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    input_args = get_args_from_command_line()

    train(args=input_args)
    print('Done')
