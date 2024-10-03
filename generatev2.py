import copy
import os
import pickle
import warnings

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from skimage.color import label2rgb
from skimage.feature import canny
from argparse import ArgumentParser

import dnnlib
from autoencoder.Autoencoderkl import GammaAutoencoderKL
from torch_utils import distributed as dist
from torch_utils import misc
from torch_utils.misc import import_module_from_path
import re


# ----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
        net, latents, class_labels=None, semantic_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=None, sigma_max=None, rho=7,
        solver='heun', discretization='edm', schedule='linear', scaling='none',
        epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
        s_churn=0, s_min=0, s_max=float('inf'), s_noise=1, **kwargs
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
            sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    if semantic_labels is not None:
        semantic_cond = True
    else:
        semantic_cond = False

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(s_churn / num_steps, np.sqrt(2) - 1) if s_min <= sigma(t_cur) <= s_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(
            t_hat) * s_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        with torch.no_grad(), torch.amp.autocast(enabled=True, device_type='cuda'):
            denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels, semantic_labels, semantic_cond)

        denoised = denoised.to(torch.float64)

        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(
            t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            with torch.no_grad(), torch.amp.autocast(enabled=True, device_type='cuda'):
                denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels, semantic_labels, semantic_cond)
            denoised = denoised.to(torch.float64)

            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(
                t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        print(size[0])
        print(len(self.generators))
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


# ----------------------------------------------------------------------------
def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_all_arguments():
    parser = ArgumentParser(description="Load config file", add_help=False)

    # General options
    parser.add_argument('--data-dir',
                        dest='data_dir',
                        type=str,
                        help='Path to the dataset')
    parser.add_argument('--results-dir',
                        dest='results_dir',
                        type=str,
                        help='Path to save the resultant images')
    parser.add_argument('--vae-cfg-pth',
                        dest='vae_cfg_pth',
                        type=str,
                        help='Path to the autoencoder config file')
    parser.add_argument('--load-vae-model-pth',
                        dest='load_vae_model_pth',
                        type=str,
                        help='Path to the autoencoder model')
    parser.add_argument('--load-diffusion-model-pth',
                        dest='load_diffusion_model_pth',
                        type=str,
                        help='Path to the diffusion model')
    parser.add_argument('--cond',
                        type=bool,
                        default=False,
                        help='Train class-conditional model')
    parser.add_argument('--semantic-cond-n-classes',
                        dest='semantic_cond_n_classes',
                        type=int,
                        default=10,
                        help='Number of classes for semantic labels')
    parser.add_argument('--diffusion-res',
                        dest='diffusion_res',
                        type=int,
                        default=128,
                        help='Diffusion network size')
    parser.add_argument('--output-res',
                        dest='output_res',
                        type=int,
                        default=256,
                        help='Output image resolution')
    parser.add_argument('--cache',
                        type=bool,
                        default=False,
                        help='Cache dataset in CPU memory')
    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help='DataLoader worker processes')
    parser.add_argument('--seeds',
                        type=str,
                        default='1-9000',
                        help='Random seeds (e.g. 1,2,5-10)',
                        action='store')
    parser.add_argument('--subdirs',
                        type=bool,
                        default=False,
                        help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class-idx',
                        dest='class_idx',
                        type=int,
                        default=None,
                        help='Class label  [default: random]')
    parser.add_argument('--batch-size',
                        dest='max_batch_size',
                        type=int,
                        default=2,
                        help='Maximum batch size')
    parser.add_argument('--steps',
                        type=int,
                        default=5,
                        help='Number of sampling steps')
    parser.add_argument('--sigma-min',
                        type=float,
                        dest='sigma_min',
                        default=None,
                        help='Lowest noise level  [default: varies]')
    parser.add_argument('--sigma-max',
                        type=float,
                        dest='sigma_max',
                        default=None,
                        help='Highest noise level  [default: varies]')
    parser.add_argument('--rho',
                        type=float,
                        default=7,
                        help='Time step exponent')
    parser.add_argument('--s-churn',
                        dest='s_churn',
                        type=float,
                        default=40,
                        help='Stochasticity strength')
    parser.add_argument('--s-min',
                        dest='s_min',
                        type=float,
                        default=0.02,
                        help='Stochastic min noise level')
    parser.add_argument('--s-max',
                        dest='s_max',
                        type=float,
                        default=100,
                        help='Stochastic max noise level')
    parser.add_argument('--s-noise',
                        dest='s_noise',
                        type=float,
                        default=1.003,
                        help='Stochastic noise inflation')
    parser.add_argument('--solver',
                        type=str,
                        choices=['euler', 'heun'],
                        default='heun',
                        help='Ablate ODE solver')
    parser.add_argument('--discretization',
                        type=str,
                        choices=['vp', 've', 'iddpm', 'edm'],
                        default='edm',
                        help='Ablate time step discretization {t_i}')
    parser.add_argument('--schedule',
                        type=str,
                        choices=['vp', 've', 'linear'],
                        default='linear',
                        help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling',
                        type=str,
                        choices=['vp', 'none'],
                        default='none',
                        help='Ablate signal scaling s(t)')

    return parser.parse_args()


def get_sampling_opts(opts):
    return {'class_idx': opts.class_idx, 'num_steps': opts.steps, 'sigma_min': opts.sigma_min,
            'sigma_max': opts.sigma_max, 'rho': opts.rho, 's_churn': opts.s_churn, 's_min': opts.s_min,
            's_max': opts.s_max, 's_noise': opts.s_noise, 'solver': opts.solver,
            'discretization': opts.discretization, 'schedule': opts.schedule, 'scaling': opts.scaling}


def main(opts):
    dist.init()
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda')

    seeds = parse_int_list(opts.seeds)
    num_batches = ((len(seeds) - 1) // (opts.max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]
    rank_batches = [x.tolist() for x in rank_batches]

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if opts.semantic_cond_n_classes > 0:
        opts.semantic_cond = True
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                                     path=opts.data_dir,
                                     data_mode='train',
                                     use_labels=opts.cond,
                                     use_semantic_labels=opts.semantic_cond,
                                     semantic_label_n_classes=opts.semantic_cond_n_classes,
                                     xflip=False,
                                     cache=opts.cache,
                                     output_resolution=opts.output_res,
                                     diffusion_resolution=opts.diffusion_res)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True,
                                         num_workers=opts.workers,
                                         prefetch_factor=1)

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{opts.load_diffusion_model_pth}"...')
    with dnnlib.util.open_url(opts.load_diffusion_model_pth, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    if opts.vae_cfg_pth is not None:
        assert os.path.isfile(opts.vae_cfg_pth), f'Path given for --vae-cfg-pth: {opts.vae_cfg_pth} not found'
        assert os.path.isfile(
            opts.load_vae_model_pth), f'Path given for --load-vae-model-pth: {opts.load_vae_model_pth} not found'

        vae_kwargs = import_module_from_path('cfg', opts.vae_cfg_pth).cfg  # Load autoencoder config
        load_vae_model_pth = opts.load_vae_model_pth
        dist.print0('Loading autoencoder...')
        autoencoder_channels = [ii * 2 ** 6 for ii in range(1, vae_kwargs.NETWORK.NUM_LAYERS + 1)]
        attention_levels = [False for _ in range(vae_kwargs.NETWORK.NUM_LAYERS - 1)] + [True]
        autoencoder = GammaAutoencoderKL(
            spatial_dims=2,
            alpha_param=vae_kwargs.NETWORK.ALPHA_PARAM,
            beta_param=vae_kwargs.NETWORK.BETA_PARAM,
            in_channels=vae_kwargs.NETWORK.IN_CHANNELS,
            out_channels=vae_kwargs.NETWORK.OUT_CHANNELS,
            num_channels=autoencoder_channels,
            latent_channels=vae_kwargs.NETWORK.LATENT_CHANNELS,
            num_res_blocks=vae_kwargs.NETWORK.NUM_RES_BLOCKS,
            norm_num_groups=vae_kwargs.NETWORK.NUM_NORM_GROUPS,
            attention_levels=attention_levels,
            use_flash_attention=vae_kwargs.NETWORK.USE_FLASH_ATTENTION,
        ).to(device)

        autoencoder.load_state_dict(torch.load(load_vae_model_pth,
                                               map_location=torch.device('cpu'),
                                               weights_only=False))
        autoencoder.eval().to(device)
        autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
        autoencoder_ddp = torch.nn.parallel.DistributedDataParallel(autoencoder,
                                                                    device_ids=[device],
                                                                    broadcast_buffers=False,
                                                                    find_unused_parameters=True)
        autoencoder_ddp = autoencoder_ddp.module if hasattr(autoencoder_ddp, "module") else autoencoder_ddp
    else:
        autoencoder_ddp = None

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{opts.results_dir}"...')
    total_img_counter = 0
    all_filenames = []

    for batch_idx, batch_seeds in enumerate(rank_batches):

        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        if batch_idx == 0 and batch_size != 0:
            dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
            dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj,
                                                   rank=dist.get_rank(),
                                                   num_replicas=dist.get_world_size(),
                                                   seed=0,
                                                   shuffle=False)
            dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj,
                                                                sampler=dataset_sampler,
                                                                batch_size=batch_size,
                                                                **data_loader_kwargs))

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size,
                             net.img_channels,
                             opts.diffusion_res,
                             opts.diffusion_res],
                            device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim,
                                                                               size=[batch_size],
                                                                               device=device)]
        if opts.class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, opts.class_idx] = 1

        if opts.semantic_cond_n_classes == 0:
            labelmaps, filenames = None, None
        else:
            __, __, labelmaps, filenames = next(dataset_iterator)
            labelmaps = torch.squeeze(labelmaps.to(device), dim=1)
            labelmaps_original = copy.deepcopy(labelmaps)

        if autoencoder_ddp is not None:
            images = ablation_sampler(net,
                                      latents,
                                      class_labels,
                                      labelmaps,
                                      randn_like=rnd.randn_like,
                                      **get_sampling_opts(opts))
            images = ((images + 1) / 2).clamp(1e-7)
            labelmaps = torch.argmax(labelmaps, dim=1).unsqueeze(1).to(torch.bool) * 1.
            images = autoencoder.decode_stage_2(images, labelmaps)

        else:
            imgs2keep = []
            for img_idx, img_seed in enumerate(batch_seeds):
                if not os.path.exists(os.path.basename(filenames[img_idx])):
                    imgs2keep.append(img_idx)

            latents = latents[imgs2keep, ...]
            labelmaps = labelmaps[imgs2keep, ...]
            filenames = [x for i, x in enumerate(filenames) if i in imgs2keep]
            batch_seeds = [x for i, x in enumerate(batch_seeds) if i in imgs2keep]
            if not batch_seeds:
                continue
            rnd = StackedRandomGenerator(device, batch_seeds)
            labelmaps_original = labelmaps_original[imgs2keep, ...]

            images = ablation_sampler(net,
                                      latents,
                                      class_labels,
                                      labelmaps,
                                      randn_like=rnd.randn_like,
                                      **get_sampling_opts(opts))
            images = ((images + 1) / 2)

        images_np = (images * 255.).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        labelmaps_original = transforms.functional.resize(labelmaps_original,
                                                          size=images.shape[-1],
                                                          interpolation=PIL.Image.NEAREST)
        labelmaps_original = torch.argmax(labelmaps_original, dim=1)
        visible_labels_out = (labelmaps_original / opts.semantic_cond_n_classes) * 255
        visible_labels_out = visible_labels_out.clip(0, 255).to(torch.uint8).cpu().numpy()

        torch.distributed.barrier()

        for rank in range(dist.get_world_size()):
            if rank == dist.get_rank():
                for img_idx, (seed, image_np, visible_labels_out, labelmaps_original) in enumerate(zip(batch_seeds,
                                                                                                       images_np,
                                                                                                       visible_labels_out,
                                                                                                       labelmaps_original)):
                    combo_img = generate_combined_imgs(visible_labels_out, np.repeat(image_np, 3, axis=-1))
                    image_dir = os.path.join(opts.results_dir,
                                             f'{seed - seed % 1000:06d}') if opts.subdirs else opts.results_dir

                    os.makedirs(os.path.join(image_dir, 'combined_images'), exist_ok=True)
                    os.makedirs(os.path.join(image_dir, 'visible_labels'), exist_ok=True)
                    os.makedirs(os.path.join(image_dir, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(image_dir, 'raw_labels'), exist_ok=True)

                    img_name = os.path.basename(filenames[img_idx])

                    combined_image_path = os.path.join(image_dir, 'combined_images', img_name)
                    visible_label_path = os.path.join(image_dir, 'visible_labels', img_name)
                    raw_label_path = os.path.join(image_dir, 'raw_labels', img_name)
                    images_path = os.path.join(image_dir, 'images', img_name)

                    if combo_img.shape[2] == 1:
                        PIL.Image.fromarray(combo_img[:, :, 0], 'L').save(combined_image_path)
                    else:
                        PIL.Image.fromarray(combo_img, 'RGB').save(combined_image_path)

                    PIL.Image.fromarray(visible_labels_out, 'L').save(visible_label_path)

                    PIL.Image.fromarray(labelmaps_original.detach().cpu().numpy(), 'L').save(raw_label_path)

                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(images_path)
                    else:
                        PIL.Image.fromarray(image_np, 'RGB').save(images_path)

                    torch.distributed.barrier()
                    all_filenames.append(filenames[img_idx])

                    total_img_counter += 1
        dist.print0(f'Batches generated: {int(batch_idx)}/{num_batches / dist.get_world_size()},'
                    f' Images {total_img_counter}/{len(seeds) / dist.get_world_size()}')
    torch.distributed.barrier()
    dist.print0('Done.')


def generate_combined_imgs(label_in_img, inference_in_img):
    if label_in_img.shape != inference_in_img.shape[:-1]:
        dist.print0('label_in_img: {}, inference_in_img: {}'.format(label_in_img.shape, inference_in_img.shape))

    overlayed_label = label2rgb(label=label_in_img,
                                image=inference_in_img,
                                bg_label=0,
                                channel_axis=-1,
                                alpha=0.2, image_alpha=1)

    overlayed_label = (overlayed_label * 255).astype('uint8')

    edges = canny(label_in_img / label_in_img.max())
    edges = np.expand_dims(edges, axis=-1)
    edges = np.concatenate((edges, edges, edges), axis=-1) * 255
    edges[:, :, 2] = 0

    overlayed_edge_label = np.copy(inference_in_img)
    overlayed_edge_label[edges == 255] = 255

    combined_imgs = np.concatenate((inference_in_img, overlayed_label, overlayed_edge_label),
                                   axis=0).astype(
        np.uint8)

    return combined_imgs


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main_args = parse_all_arguments()
    main(main_args)

# ----------------------------------------------------------------------------
