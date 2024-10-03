import os
import warnings
from argparse import ArgumentParser

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from monai.losses import PerceptualLoss
import torch._dynamo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training.epoch_cycles import train_autoencoder_cycle, evaluate_autoencoder_cycle
from training.dataset import CamusVAEDataset
from torch_utils import distributed as dist
from autoencoder.Autoencoderkl import GammaAutoencoderKL
from torch_utils.misc import import_module_from_path

torch._dynamo.config.suppress_errors = True

torchvision.disable_beta_transforms_warning()

warnings.filterwarnings('ignore',
                        'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.


def parse_all_arguments():
    if dist.get_rank() == 0:
        config_parser = ArgumentParser(description="Load config file", add_help=False)
        config_parser.add_argument('--config',
                                   type=str,
                                   help="Path to the config file")
        args, remaining_argv = config_parser.parse_known_args()
        if args.config:
            cfg = import_module_from_path('cfg', args.config).cfg
        else:
            cfg = {}
    torch.distributed.barrier()
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
    parser.add_argument('--alpha-param',
                        dest='alpha_param',
                        help='alpha parameter for gamma distribution',
                        default=cfg.NETWORK.ALPHA_PARAM,
                        type=float)
    parser.add_argument('--beta-param',
                        dest='beta_param',
                        help='beta parameter for gamma distribution',
                        default=cfg.NETWORK.BETA_PARAM,
                        type=float)
    parser.add_argument('--num-layers',
                        dest='num_layers',
                        help='number of layers in autoencoder',
                        default=cfg.NETWORK.NUM_LAYERS,
                        type=int)
    parser.add_argument('--in-channels',
                        dest='in_channels',
                        help='Number of input image channels',
                        default=cfg.NETWORK.IN_CHANNELS,
                        type=int)
    parser.add_argument('--out-channels',
                        dest='out_channels',
                        help='Number of output image channels',
                        default=cfg.NETWORK.OUT_CHANNELS,
                        type=int)
    parser.add_argument('--latent-channels',
                        dest='latent_channels',
                        help='Number of latent image channels',
                        default=cfg.NETWORK.LATENT_CHANNELS,
                        type=int)
    parser.add_argument('--num-res-blocks',
                        dest='num_res_blocks',
                        help='Number of residual blocks in autoencoder',
                        default=cfg.NETWORK.NUM_RES_BLOCKS,
                        type=int)
    parser.add_argument('--num-norm-groups',
                        dest='num_norm_groups',
                        help='Number of normalization groups in layernorm',
                        default=cfg.NETWORK.NUM_NORM_GROUPS,
                        type=int)
    parser.add_argument('--use-flash-attention',
                        dest='use_flash_attention',
                        help='Whether to use flash attention',
                        default=cfg.NETWORK.USE_FLASH_ATTENTION,
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
    parser.add_argument('--amp',
                        help='Whether to use torch automatic mixed precision',
                        default=cfg.CONST.AMP,
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
    parser.add_argument('--validation-interval',
                        dest='validation_interval',
                        help='number of epochs to run training for',
                        default=cfg.TRAIN.NUM_EPOCHS,
                        type=int)
    parser.add_argument('--lr',
                        dest='lr',
                        help='learning rate',
                        default=cfg.TRAIN.LR,
                        type=float)
    parser.add_argument('--min-lr',
                        dest='min_lr',
                        help='lr will decay to this value over training cycle using cosine annealing lr schedule',
                        default=cfg.TRAIN.MIN_LR,
                        type=float)
    parser.add_argument('--momentum',
                        help='momentum value for SGD optimiser',
                        default=cfg.TRAIN.MOMENTUM,
                        type=float)
    parser.add_argument('--nesterov',
                        help='Whether to use nesterov momentum in SGD optimiser',
                        default=cfg.TRAIN.NESTEROV,
                        type=bool)
    parser.add_argument('--shuffle',
                        help='Whether to shuffle training data',
                        default=cfg.TRAIN.SHUFFLE,
                        type=bool)
    parser.add_argument('--seed',
                        help='random seed for initialisation',
                        default=cfg.TRAIN.SEED)
    ############################################ Loss Parameters ############################################
    parser.add_argument('--perceptual-weight',
                        dest='perceptual_weight',
                        help='Weight for perceptual loss',
                        default=cfg.LOSS.PERCEPTUAL_WEIGHT,
                        type=float)
    parser.add_argument('--kl-weight',
                        dest='kl_weight',
                        help='Weight for Kullback–Leibler divergence loss',
                        default=cfg.LOSS.KL_WEIGHT,
                        type=float)
    parser.add_argument('--mse-weight',
                        dest='mse_weight',
                        help='Weight for Mean Squared Error reconstruction loss',
                        default=cfg.LOSS.MSE_WEIGHT,
                        type=float)
    parser.add_argument('--mse-latent-weight',
                        dest='mse_latent_weight',
                        help='Weight for Mean Squared Error reconstruction loss on latent image',
                        default=cfg.LOSS.MSE_LATENT_WEIGHT,
                        type=float)

    return parser.parse_args(remaining_argv)


def main(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logs_path = os.path.join(args.results_dir, 'logs')
    if rank == 0:
        os.makedirs(logs_path, exist_ok=True)
        train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
        val_writer = SummaryWriter(os.path.join(logs_path, 'val'))
        latent_res = args.image_size // args.num_layers
        trained_g_path = os.path.join(args.results_dir, "autoencoder_" + str(latent_res) + ".pth")
        dist.print0(
            f'Input image of size of: [{args.image_size}] and number of layers: [{args.num_layers}] \n'
            f'Results in a latent size of: [{latent_res}]')

    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_transforms = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size)),
         transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(degrees=10,
                                   interpolation=transforms.InterpolationMode.NEAREST,
                                   fill=0),
         transforms.RandomAffine(degrees=0,
                                 translate=(0.1, 0.1),
                                 scale=(1, 1),
                                 interpolation=transforms.InterpolationMode.NEAREST,
                                 fill=0),
         transforms.RandomAffine(degrees=0,
                                 translate=(0, 0),
                                 scale=(0.8, 1.2),
                                 interpolation=transforms.InterpolationMode.NEAREST, fill=0), ])

    val_transforms = transforms.Compose([transforms.Resize((args.image_size, args.image_size))])

    autoencoder_channels = [ii * 2 ** 6 for ii in range(1, args.num_layers + 1)]
    attention_levels = [False for _ in range(args.num_layers - 1)] + [True]

    model = GammaAutoencoderKL(
        alpha_param=args.alpha_param,
        beta_param=args.beta_param,
        spatial_dims=2,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_channels=autoencoder_channels,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        norm_num_groups=args.num_norm_groups,
        attention_levels=attention_levels,
        use_flash_attention=args.use_flash_attention,
    ).to(rank)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    torch.distributed.barrier()
    if rank == 0:
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        perceptual_loss = PerceptualLoss(spatial_dims=2,
                                         network_type="radimagenet_resnet50",
                                         is_fake_3d=False).to(rank)
        if args.compile:
            try:
                model = torch.compile(model, dynamic=False)
            except Exception as e:
                print('Failed to compile mode, continuing without compiled model')

    torch.distributed.barrier()
    if rank != 0:
        perceptual_loss = PerceptualLoss(spatial_dims=2,
                                         network_type="radimagenet_resnet50",
                                         is_fake_3d=False).to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[rank],
                                                      find_unused_parameters=True)

    train_dataset = CamusVAEDataset(args.data_dir, 'training', data_type=args.data_type)
    train_sampler = DistributedSampler(train_dataset, rank=rank)

    val_dataset = CamusVAEDataset(args.data_dir, 'validation', data_type=args.data_type)
    val_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              persistent_workers=True,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            persistent_workers=True,
                            sampler=val_sampler)

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr * world_size,
                                momentum=args.momentum,
                                nesterov=args.nesterov)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)

    best_val_loss = np.inf

    scaler_g = torch.amp.GradScaler('cuda')

    def loss_func(recon_img, target_img):
        return torch.squeeze(F.mse_loss(recon_img.float(), target_img.float()))

    torch.distributed.barrier()
    for epoch in range(args.num_epochs):

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        total_loss_train, recons_loss_train, kl_loss_train, recons_loss_latent_train = train_autoencoder_cycle(args,
                                                                                                               epoch,
                                                                                                               model,
                                                                                                               perceptual_loss,
                                                                                                               train_loader,
                                                                                                               train_transforms,
                                                                                                               loss_func,
                                                                                                               optimizer,
                                                                                                               lr_scheduler,
                                                                                                               scaler_g,
                                                                                                               rank,
                                                                                                               train_writer)

        if epoch % args.validation_interval == 0:
            total_loss_val, recons_loss_val, kl_loss_val = evaluate_autoencoder_cycle(args,
                                                                                      epoch,
                                                                                      model,
                                                                                      perceptual_loss,
                                                                                      val_loader,
                                                                                      val_transforms,
                                                                                      loss_func,
                                                                                      rank,
                                                                                      val_writer)
            dist.print0(f"Epoch {epoch} val_loss: {total_loss_val}")
            if rank == 0:
                val_writer.add_scalar('loss', total_loss_val, epoch)
                if total_loss_val < best_val_loss and rank == 0:
                    best_val_loss = total_loss_val
                    torch.save(model.module.state_dict(), trained_g_path)
                    dist.print0("===== saved best model ======")
                    val_writer.add_scalar('best_val_loss', best_val_loss, epoch)

        if rank == 0:
            train_writer.add_scalar('loss', total_loss_train, epoch)
            train_writer.add_scalar('recons_loss', recons_loss_train, epoch)
            train_writer.add_scalar('kl_loss', kl_loss_train, epoch)
            train_writer.add_scalar('recons_latent_loss', recons_loss_latent_train, epoch)

    torch.distributed.destroy_process_group()
    return


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    main_args = parse_all_arguments()
    main(main_args)
