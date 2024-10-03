import copy
import json
import os
import time

import numpy as np
import torch

import dnnlib
from autoencoder.Autoencoderkl import GammaAutoencoderKL
from torch_utils import distributed as dist
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.network_utils import append_diffusion_train_output, diffusion_load_from_pkl, \
    diffusion_load_from_state_dump, \
    pkl_dump_best_iter
from training.epoch_cycles import train_diffusion_cycle, validate_diffusion_cycle


def loop(
        run_dir='.',  # Output directory.
        dataset_kwargs=None,  # Options for training set.
        dataset_kwargs_val=None,  # Options for validation set.
        val_freq=None,  # How often to run validation.
        data_loader_kwargs=None,  # Options for torch.torch_utils.data.DataLoader.
        network_kwargs=None,  # Options for model and preconditioning.
        vae_kwargs=None,  # Options for autoencoder.
        load_vae_model_pth=None,  # Path to autoencoder model.
        loss_kwargs=None,  # Options for loss function.
        optimizer_kwargs=None,  # Options for optimizer.
        augment_kwargs=None,  # Options for training augmentation pipeline, None = disable.
        augment_kwargs_val=None,  # Options for validation augmentation pipeline, None = disable.
        seed=0,  # Global random seed.
        batch_size=512,  # Total batch size for one training iteration.
        batch_gpu=None,  # Limit batch size per GPU, None = no limit.
        total_kimg=200000,  # Training duration, measured in thousands of training images.
        ema_halflife_kimg=5,  # Half-life of the exponential moving average (EMA) of model weights.
        ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
        loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
        kimg_per_tick=50,  # Interval of progress prints.
        snapshot_ticks=50,  # How often to save network snapshots, None = disable.
        state_dump_ticks=500,  # How often to dump training state, None = disable.
        resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
        resume_state_dump=None,  # Start from the given training state, None = reset training state.
        resume_kimg=0,  # Start from the given training progress.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        device=torch.device('cuda'),
):
    # Initialize.
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if dataset_kwargs_val is None:
        dataset_kwargs_val = {}
    if data_loader_kwargs is None:
        data_loader_kwargs = {}
    if network_kwargs is None:
        network_kwargs = {}
    if loss_kwargs is None:
        loss_kwargs = {}
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Select batch size per GPU.

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size(),
                                           seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj,
                                                        sampler=dataset_sampler,
                                                        batch_size=batch_gpu,
                                                        **data_loader_kwargs))
    if dataset_kwargs_val is not None:
        dataset_obj_val = dnnlib.util.construct_class_by_name(**dataset_kwargs_val)
        dataset_sampler_val = misc.InfiniteSampler(dataset=dataset_obj_val,
                                                   rank=dist.get_rank(),
                                                   num_replicas=dist.get_world_size(),
                                                   seed=seed)
        dataset_iterator_val = iter(torch.utils.data.DataLoader(dataset=dataset_obj_val,
                                                                sampler=dataset_sampler_val,
                                                                batch_size=batch_gpu,
                                                                **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=int(dataset_kwargs.diffusion_resolution),
                            img_channels=1,
                            label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss

    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(),
                                                    **optimizer_kwargs)  # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(
        **augment_kwargs) if augment_kwargs is not None else None  # training.augment.AugmentPipe
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        diffusion_load_from_pkl(resume_pkl, net, ema)
    if resume_state_dump:
        diffusion_load_from_state_dump(resume_state_dump, net, optimizer)

    if vae_kwargs is not None:
        dist.print0('Loading autoencoder...')
        autoencoder_channels = [ii * 2 ** 6 for ii in range(1, vae_kwargs.NETWORK.NUM_LAYERS + 1)]
        attention_levels = [False for _ in range(vae_kwargs.NETWORK.NUM_LAYERS - 1)] + [True]
        autoencoder = GammaAutoencoderKL(
            spatial_dims=2,
            alpha_param=vae_kwargs.alpha_param,
            beta_param=vae_kwargs.beta_param,
            in_channels=vae_kwargs.NETWORK.IN_CHANNELS,
            out_channels=vae_kwargs.NETWORK.OUT_CHANNELS,
            num_channels=autoencoder_channels,
            latent_channels=vae_kwargs.NETWORK.LATENT_CHANNELS,
            num_res_blocks=vae_kwargs.NETWORK.NUM_RES_BLOCKS,
            norm_num_groups=vae_kwargs.NETWORK.NUM_NORM_GROUPS,
            attention_levels=attention_levels,
            use_flash_attention=vae_kwargs.NETWORK.USE_FLASH_ATTENTION,
        ).to(device)

        autoencoder.load_state_dict(
            torch.load(load_vae_model_pth, map_location=torch.device('cpu'), weights_only=False))
        autoencoder.eval().to(device)
        autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
        autoencoder_ddp = torch.nn.parallel.DistributedDataParallel(autoencoder,
                                                                    device_ids=[device],
                                                                    broadcast_buffers=False,
                                                                    find_unused_parameters=True)
        autoencoder_ddp = autoencoder_ddp.module if hasattr(autoencoder_ddp, "module") else autoencoder_ddp
    else:
        autoencoder_ddp = None

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        train_diffusion_cycle(ddp, net, dataset_iterator, device, loss_fn, optimizer, ema, training_stats, loss_scaling,
                              num_accumulation_rounds, batch_gpu_total, vae_kwargs, augment_pipe, network_kwargs,
                              dataset_kwargs, autoencoder_ddp, ema_halflife_kimg, ema_rampup_ratio, cur_nimg,
                              batch_size)

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        fields, tick_end_time = append_diffusion_train_output(training_stats,
                                                              cur_tick,
                                                              cur_nimg,
                                                              tick_start_nimg,
                                                              total_kimg,
                                                              start_time,
                                                              tick_start_time,
                                                              maintenance_time, device)
        torch.cuda.reset_peak_memory_stats()

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            savepath = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            pkl_dump_best_iter(ema, loss_fn, augment_pipe, dataset_kwargs, savepath)

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (
                done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()),
                       os.path.join(run_dir, f'training-state-{cur_nimg // 1000:06d}.pt'))

        training_stats.default_collector.update()
        fields += [f"Train Loss {training_stats.default_collector.as_dict()['Loss/loss']['mean']:<7.4f}"]

        if dataset_kwargs_val is not None and (val_freq is not None) and (cur_tick % val_freq == 0):
            validate_diffusion_cycle(ddp_net=ddp,
                                     autoencoder_ddp=autoencoder_ddp,
                                     dataset_iterator_val=dataset_iterator_val,
                                     loss_fn=loss_fn,
                                     network_kwargs=network_kwargs,
                                     dataset_kwargs=dataset_kwargs,
                                     total_kimg_val=len(dataset_obj_val) / 1000,
                                     num_accumulation_rounds=num_accumulation_rounds,
                                     batch_size=batch_size,
                                     input_training_stats=training_stats,
                                     device=device)
            training_stats.val_collector.update()
            if cur_tick == 0:
                training_stats.report0('best_val_kimg', cur_nimg // 1000)
                training_stats.report0('best_val_loss',
                                       training_stats.val_collector.as_dict()['Val_loss/val_loss']['mean'])

            if training_stats.val_collector.as_dict()['Val_loss/val_loss']['mean'] < \
                    training_stats.val_collector.as_dict()['best_val_loss']['mean']:
                training_stats.report('best_val_loss',
                                      training_stats.val_collector.as_dict()['Val_loss/val_loss']['mean'])
                training_stats.report('best_val_kimg', cur_nimg // 1000)
                training_stats.val_collector.update()

            savepath = os.path.join(run_dir, f'best_val_iter.pkl')
            pkl_dump_best_iter(ema, loss_fn, augment_pipe, dataset_kwargs, savepath)

            fields += [f"Val Loss {training_stats.val_collector.as_dict()['Val_loss/val_loss']['mean']:<7.4f}"]
            fields += [f"Best Val Loss {training_stats.val_collector.as_dict()['best_val_loss']['mean']:<7.4f}"]
            fields += [f"Best Val kimg {training_stats.val_collector.as_dict()['best_val_kimg']['mean']:<9.1f}"]

        dist.print0(' '.join(fields))

        # Update logs.
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(
                json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0()
    dist.print0('Exiting...')
