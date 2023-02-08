# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch.distributions import Beta
import glob
from torch_utils import misc
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0, alpha=0., pfgm=False,
    pfgmpp=False, align=False, D=128, align_precond=False,
):

    if pfgm:
        #print("rho:", rho)
        # Adjust noise levels based on what's supported by the network.
        N = net.img_channels * net.img_resolution * net.img_resolution
        r_min = 0.55 / np.sqrt(N / (D - 2 - 1))
        r_max = 2500 / np.sqrt(N / (D - 2 - 1))

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (r_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    r_min ** (1 / rho) - r_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0


        # samples_norm = torch.sqrt(latents) * r_max
        # samples_norm = samples_norm.view(len(samples_norm), -1)
        # # Uniformly sample the angle direction
        # gaussian = torch.randn(len(latents), N).to(samples_norm.device)
        # unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # # Radius times the angle direction
        # init_samples = unit_gaussian * samples_norm
        # latents = init_samples.reshape((len(latents), net.img_channels, net.img_resolution, net.img_resolution))
        x_next = latents.to(torch.float64)

        # Main sampling loop.
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            t_hat = net.round_sigma(t_cur)
            x_hat = x_cur

            # Euler step.
            x_drift, z_drift = net(x_hat, t_hat, class_labels)
            x_drift = x_drift.view(len(x_drift), -1).to(torch.float64)
            z_drift = z_drift.to(torch.float64) * np.sqrt(D)
            # Predicted normalized Poisson field
            v = torch.cat([x_drift, z_drift[:, None]], dim=1)
            dt_dz = 1 / (v[:, -1] + 1e-5)
            dx_dt = v[:, :-1].view(len(x_drift), net.img_channels,
                                   net.img_resolution,
                                   net.img_resolution)
            dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x_hat.size()[1:])))
            d_cur = dx_dz
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                x_drift_new, z_drift_new = net(x_next, t_next, class_labels)
                x_drift_new = x_drift_new.view(len(x_drift_new), -1).to(torch.float64)
                z_drift_new = z_drift_new.to(torch.float64) * np.sqrt(D)
                # Predicted normalized Poisson field
                v_new = torch.cat([x_drift_new, z_drift_new[:, None]], dim=1)
                dt_dz_new = 1 / (v_new[:, -1] + 1e-5)
                dx_dt_new = v_new[:, :-1].view(len(x_drift_new), net.img_channels,
                                       net.img_resolution,
                                       net.img_resolution)
                dx_dz_new = dx_dt_new * dt_dz_new.view(-1, *([1] * len(x_next.size()[1:])))
                d_prime = dx_dz_new
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    else:
        N = net.img_channels * net.img_resolution * net.img_resolution
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        if align:
            sigma_min *= np.sqrt(1 + N/D)
            sigma_max *= np.sqrt(1 + N/D)

        #print("sigma max:", sigma_max, "sigma min:", sigma_min)
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        #t_steps = t_steps[:-2]
        if pfgmpp:
            x_next = latents.to(torch.float64)
        else:
            x_next = latents.to(torch.float64) * t_steps[0]
        # Main sampling loop.
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1

            x_cur = x_next

            gaussian = torch.randn((len(x_cur), N)).to(x_cur.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            unit_gaussian = unit_gaussian.view_as(x_cur)
            #if i < 15:
            x_cur += torch.randn_like(x_cur) * t_cur * alpha
            # radius = x_cur.view(len(x_cur), -1).norm(p=2, dim=1) * alpha
            # radius = radius.reshape((-1, 1, 1, 1))
            # x_cur += unit_gaussian * radius

            # norm = x_cur.view(len(x_cur), -1).norm(p=2, dim=1)/(t_cur * np.sqrt(N))
            # print(f"i:{i}, t cur:{t_cur:.3f}, norm/\sigma * sqrt({N}):",
            #      f"max: {max(norm):.3f}, min: {min(norm):.3f}")

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # Euler step.
            if align_precond:
                t_old = t_hat / np.sqrt(1 + N/D)
                #print(t_old, t_hat)
                denoised = net(x_hat, t_hat, class_labels, sigma_old=t_old).to(torch.float64)
            else:
                denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                if align_precond:
                    t_old = t_next / np.sqrt(1 + N/D)
                    denoised = net(x_next, t_next, class_labels, sigma_old=t_old).to(torch.float64)
                else:
                    denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    #print("mean final norm:", x_next.reshape((len(x_next), -1)).norm(p=2, dim=1).mean(), x_next.shape)
    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
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
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

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
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def rand_beta_prime(self, size, N=3072, D=128, **kwargs):
        # sample from beta_prime (N/2, D/2)
        # print(f"N:{N}, D:{D}")
        assert size[0] == len(self.seeds)
        latent_list = []
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        for seed in self.seeds:
            torch.manual_seed(seed)
            sample_norm = beta_gen.sample().to(kwargs['device']).double()
            # inverse beta distribution
            inverse_beta = sample_norm / (1-sample_norm)
            if kwargs['pfgmpp']:
                #S_max = 200 if D==128 else 80
                S_max = 80
                sample_norm = torch.sqrt(inverse_beta) * S_max * np.sqrt(D)

            gaussian = torch.randn(N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            init_sample = unit_gaussian * sample_norm
            latent_list.append(init_sample.reshape((1, *size[1:])))

        latent = torch.cat(latent_list, dim=0)
        return latent

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--save_images',             help='only save a batch images for grid visualization',                     is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=0, show_default=True)
#@click.option('--alpha', 'alpha',          help='noise norm', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--ckpt', 'ckpt',      help='begin ckpt', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--resume', 'resume',      help='resume ckpt', metavar='INT',                          type=int, default=None, show_default=True)
@click.option('--end_ckpt', 'end_ckpt',      help='end ckpt', metavar='INT',                          type=int, default=100000000, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--edm',          help='load edm model', metavar='BOOL',              type=bool, default=False, show_default=True)

@click.option('--pfgm',          help='Train PFGM', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--pfgmpp',          help='Train pfgmpp', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--align',          help='Align', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--align_precond',          help='Align', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--aug_dim',             help='additional dimension', metavar='INT',                            type=click.IntRange(min=2), default=128, show_default=True)

def main(ckpt, end_ckpt, outdir, subdirs, seeds, class_idx, max_batch_size, save_images, pfgm, pfgmpp, align, aug_dim, edm, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    if not edm:
        stats = glob.glob(os.path.join(outdir, "training-state-*.pt"))
    else:
        stats = glob.glob(os.path.join(outdir, "network-snapshot-*.pkl"))
    done_list = []

    step_list = [10, 12, 14, 16, 18, 20, 22, 24]
    #step_list = [20, 22, 24, 26, 28, 30]
    for ckpt_dir in stats:
        # Load network.
        dist.print0(f'Loading network from "{ckpt_dir}"...')
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        # with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        #     net = pickle.load(f)['ema'].to(device)

        if edm:
            with dnnlib.util.open_url(ckpt_dir, verbose=(dist.get_rank() == 0)) as f:
                net = pickle.load(f)['ema'].to(device)
            ckpt_num = 0
        else:
            ckpt_num = int(ckpt_dir[-9:-3])
            data = torch.load(ckpt_dir, map_location=torch.device('cpu'))
            net = data['ema'].to(device)
            assert net.D == aug_dim

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        for steps in step_list:

            if seeds[-1] > 49999 and seeds[-1] <= 99999:
                temp_dir = os.path.join(outdir, f'ckpt_2_{ckpt_num:06d}_steps_{steps}')
            elif seeds[-1] > 99999:
                temp_dir = os.path.join(outdir, f'ckpt_3_{ckpt_num:06d}_steps_{steps}')
            else:
                temp_dir = os.path.join(outdir, f'ckpt_{ckpt_num:06d}_steps_{steps}')

            if not edm:
                if ckpt_num < ckpt or ckpt_num > end_ckpt or ckpt_num in done_list:
                    continue
            if os.path.exists(temp_dir) and not save_images:
                continue

            # Loop over batches.
            dist.print0(f'Generating {len(seeds)} images to "{temp_dir}"...')
            for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
                torch.distributed.barrier()
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                N = net.img_channels * net.img_resolution * net.img_resolution
                # Pick latents and labels.
                rnd = StackedRandomGenerator(device, batch_seeds)
                if pfgm or pfgmpp:
                    latents = rnd.rand_beta_prime(
                        [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
                        N=N,
                        D=aug_dim,
                        pfgm=pfgm,
                        pfgmpp=pfgmpp,
                        align=align,
                        device=device)
                else:
                    latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution],
                                        device=device)
                class_labels = None
                if net.label_dim:
                    class_labels = torch.eye(net.label_dim, device=device)[
                        rnd.randint(net.label_dim, size=[batch_size], device=device)]
                if class_idx is not None:
                    class_labels[:, :] = 0
                    class_labels[:, class_idx] = 1

                # Generate images.
                sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
                sampler_kwargs['num_steps'] = steps
                have_ablation_kwargs = any(
                    x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
                sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
                images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like,
                                    pfgm=pfgm, pfgmpp=pfgmpp, D=aug_dim, align=align,  **sampler_kwargs)

                # Save images.
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

                for seed, image_np in zip(batch_seeds, images_np):

                    # image_dir = os.path.join(temp_dir, f'{seed - seed % 1000:06d}') if subdirs else outdir
                    image_dir = os.path.join(temp_dir, f'{seed - seed % 1000:06d}')
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{seed:06d}.png')
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
            # Done.
            torch.distributed.barrier()

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
