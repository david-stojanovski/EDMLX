import json
import os
import re
import warnings
from argparse import ArgumentParser

import click
import torch
import torchvision

import dnnlib
from torch_utils import distributed as dist
from torch_utils.misc import import_module_from_path
from training import diffusion_loop

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
            cfg_diffusion = import_module_from_path('cfg', args.config).cfg
        else:
            cfg_diffusion = {}

    torch.distributed.barrier()
    parser = ArgumentParser(
        parents=[config_parser],
        description="Script with configurable defaults"
    )

    # General options
    parser.add_argument('--data-dir',
                        dest='data_dir',
                        type=str,
                        default=cfg_diffusion.DATASET.DATA_DIR,
                        help='Path to the dataset')
    parser.add_argument('--results-dir',
                        dest='results_dir',
                        type=str,
                        default=cfg_diffusion.DATASET.RESULTS_DIR,
                        help='Where to save the results')
    parser.add_argument('--vae-cfg-pth',
                        dest='vae_cfg_pth',
                        type=str,
                        default=cfg_diffusion.DATASET.VAE_CFG_PTH,
                        help='Path to the autoencoder model')
    parser.add_argument('--load-vae-model-pth',
                        dest='load_vae_model_pth',
                        type=str,
                        default=cfg_diffusion.DATASET.LOAD_VAE_MODEL_PTH,
                        help='Path to the autoencoder model')
    parser.add_argument('--output-res',
                        dest='output_res',
                        type=int,
                        default=cfg_diffusion.NETWORK.OUPUT_RES,
                        help='Image network size')
    parser.add_argument('--diffusion-res',
                        dest='diffusion_res',
                        type=int,
                        default=cfg_diffusion.NETWORK.DIFFUSION_RES,
                        help='Diffusion network size')
    parser.add_argument('--cond',
                        type=bool,
                        default=cfg_diffusion.NETWORK.COND,
                        help='Train class-conditional model')
    parser.add_argument('--semantic-cond-n-classes',
                        dest='semantic_cond_n_classes',
                        type=int,
                        default=cfg_diffusion.NETWORK.SEMANTIC_COND_N_CLASSES,
                        help='Number of classes for semantic labels')
    parser.add_argument('--arch',
                        type=str,
                        choices=['ddpmpp', 'ncsnpp', 'adm'],
                        default=cfg_diffusion.NETWORK.ARCH,
                        help='Network architecture')
    parser.add_argument('--precond',
                        type=str,
                        choices=['vp', 've', 'edm'],
                        default='edm',
                        help='Preconditioning & loss function')
    parser.add_argument('--duration',
                        type=float,
                        default=cfg_diffusion.TRAIN.DURATION,
                        help='Training duration in MIMG')
    parser.add_argument('--val-freq',
                        dest='val_freq',
                        type=float,
                        default=cfg_diffusion.TRAIN.VAL_FREQ,
                        help='Validation frequency in KIMG')
    parser.add_argument('--batch',
                        type=int,
                        default=cfg_diffusion.CONST.BATCH_SIZE,
                        help='Total batch size')
    parser.add_argument('--batch-gpu',
                        dest='batch_gpu',
                        type=int,
                        default=cfg_diffusion.CONST.BATCH_GPU,
                        help='Batch size per GPU')
    parser.add_argument('--cbase',
                        type=int,
                        default=cfg_diffusion.NETWORK.CBASE,
                        help='Channel multiplier')
    parser.add_argument('--cres',
                        type=str,
                        default=cfg_diffusion.NETWORK.CRES,
                        help='Channels per resolution',
                        action='store',
                        nargs='+')
    parser.add_argument('--lr',
                        type=float,
                        default=cfg_diffusion.TRAIN.LR,
                        help='Learning rate')
    parser.add_argument('--ema',
                        type=float,
                        default=cfg_diffusion.TRAIN.EMA,
                        help='EMA half-life in MIMG')
    parser.add_argument('--dropout',
                        type=float,
                        default=cfg_diffusion.TRAIN.DROPOUT,
                        help='Dropout probability')
    parser.add_argument('--augment',
                        type=float,
                        default=cfg_diffusion.TRAIN.AUGMENT,
                        help='Augment probability')
    parser.add_argument('--use-ugment-v2',
                        dest='use_augment_v2',
                        type=float,
                        default=cfg_diffusion.TRAIN.USE_AUGMENT_V2,
                        help='Augment probability')
    parser.add_argument('--xflip',
                        type=bool,
                        default=False,
                        help='Enable dataset x-flips')
    parser.add_argument('--amp',
                        type=bool,
                        default=cfg_diffusion.CONST.AMP,
                        help='Enable mixed-precision training')
    parser.add_argument('--ls',
                        type=float,
                        default=cfg_diffusion.TRAIN.LS,
                        help='Loss scaling')
    parser.add_argument('--bench',
                        type=bool,
                        default=cfg_diffusion.CONST.BENCH if torch.cuda.is_available() else False,
                        help='Enable cuDNN benchmarking')
    parser.add_argument('--cache',
                        type=bool,
                        default=cfg_diffusion.CONST.CACHE,
                        help='Cache dataset in CPU memory')
    parser.add_argument('--workers',
                        type=int,
                        default=cfg_diffusion.CONST.NUM_WORKERS,
                        help='DataLoader worker processes')
    # I/O-related options
    parser.add_argument('--nosubdir',
                        action='store_true',
                        default=cfg_diffusion.IO.NOSUBDIR,
                        help='Do not create a subdirectory for results')
    parser.add_argument('--tick',
                        type=int,
                        default=cfg_diffusion.IO.TICK,
                        help='How often to print progress in KIMG')
    parser.add_argument('--snap',
                        type=int,
                        default=cfg_diffusion.IO.SNAP,
                        help='How often to save snapshots in TICKS')
    parser.add_argument('--dump',
                        type=int,
                        default=cfg_diffusion.IO.DUMP,
                        help='How often to dump state in TICKS')
    parser.add_argument('--seed',
                        type=int,
                        default=cfg_diffusion.CONST.SEED,
                        help='Random seed')
    parser.add_argument('--transfer',
                        type=str,
                        default=cfg_diffusion.DATASET.LOAD_PKL_PTH,
                        help='Transfer learning from network pickle')
    parser.add_argument('--resume',
                        type=str,
                        default=cfg_diffusion.DATASET.LOAD_MODEL_PTH,
                        help='Resume from previous training state')
    parser.add_argument('--desc',
                        type=str,
                        default=cfg_diffusion.IO.DESC,
                        help='Resume from previous training state')
    parser.add_argument('--dry-run',
                        dest='dry_run',
                        default=cfg_diffusion.IO.DRY_RUN,
                        action='store_true',
                        help='Print training options and exit')

    return parser.parse_args(remaining_argv)


def main(opts):
    if opts.semantic_cond_n_classes > 0:
        opts.semantic_cond = True

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                                       path=opts.data_dir,
                                       data_mode='train',
                                       use_labels=opts.cond,
                                       use_semantic_labels=opts.semantic_cond,
                                       semantic_label_n_classes=opts.semantic_cond_n_classes,
                                       xflip=opts.xflip,
                                       cache=opts.cache,
                                       output_resolution=opts.output_res,
                                       diffusion_resolution=opts.diffusion_res)

    if opts.vae_cfg_pth is not None:
        assert os.path.isfile(opts.vae_cfg_pth), f'Path given for --vae-cfg-pth: {opts.vae_cfg_pth} not found'

        c.vae_kwargs = import_module_from_path('cfg',
                                               opts.vae_cfg_pth).cfg  # Load autoencoder config in perhaps a messy way
        c.load_vae_model_pth = opts.load_vae_model_pth
    else:
        c.vae_kwargs = None

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW',
                                         lr=opts.lr,
                                         betas=[0.9, 0.999],
                                         eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.max_size = len(dataset_obj)
        assert opts.val_freq >= 0
        if opts.val_freq > 0:
            c.dataset_kwargs_val = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                                                   path=opts.data_dir,
                                                   data_mode='val',
                                                   use_labels=opts.cond,
                                                   use_semantic_labels=opts.semantic_cond,
                                                   semantic_label_n_classes=opts.semantic_cond_n_classes,
                                                   xflip=False,
                                                   cache=opts.cache,
                                                   output_resolution=opts.output_res,
                                                   diffusion_resolution=opts.diffusion_res)
            dataset_obj_val = dnnlib.util.construct_class_by_name(**c.dataset_kwargs_val)
            c.dataset_kwargs_val.max_size = len(dataset_obj_val)
        else:
            c.dataset_kwargs_val.val_freq = None
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj  # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet',
                                embedding_type='positional',
                                encoder_type='standard',
                                decoder_type='standard',
                                semantic_cond=opts.semantic_cond,
                                semantic_label_nc=opts.semantic_cond_n_classes)
        c.network_kwargs.update(channel_mult_noise=1,
                                resample_filter=[1, 1],
                                model_channels=128,
                                channel_mult=[2, 2, 2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet',
                                embedding_type='fourier',
                                encoder_type='residual',
                                decoder_type='standard',
                                semantic_cond=opts.semantic_cond,
                                semantic_label_nc=opts.semantic_cond_n_classes)
        c.network_kwargs.update(channel_mult_noise=2,
                                resample_filter=[1, 3, 3, 1],
                                model_channels=128,
                                channel_mult=[2, 2, 2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet',
                                model_channels=192,
                                channel_mult=[1, 2, 3, 4],
                                semantic_cond=opts.semantic_cond,
                                semantic_label_nc=opts.semantic_cond_n_classes)

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'

    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        if opts.use_augment_v2:
            aug_class_name = 'training.augment.AugmentPipeV2'
        else:
            aug_class_name = 'training.augment.AugmentPipe'
        c.augment_kwargs = dnnlib.EasyDict(class_name=aug_class_name, p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        if opts.cond:
            c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.amp)

    # Training options.
    c.total_kimg = max((opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(val_freq=opts.val_freq)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    semantic_cond_str = 'semantic_cond' if c.dataset_kwargs.use_semantic_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{semantic_cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.results_dir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.results_dir):
            prev_run_dirs = [x for x in os.listdir(opts.results_dir) if
                             os.path.isdir(os.path.join(opts.results_dir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.results_dir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Semantic conditional:    {c.dataset_kwargs.use_semantic_labels}')
    dist.print0(f'Validation frequency:    {opts.val_freq} kimg')
    dist.print0(f'Network architecture:    {opts.arch}')
    if opts.vae_cfg_pth is not None:
        dist.print0(f'Latent Encoder:          {opts.vae_cfg_pth}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    diffusion_loop.loop(**c)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    main_args = parse_all_arguments()
    main(main_args)
# ----------------------------------------------------------------------------
