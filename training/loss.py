# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import numpy as np
from scipy.stats import betaprime
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, net, images, labels, augment_pipe=None, stf=False, pfgmpp=False, ref_images=None):
        if pfgmpp:
            rnd_uniform = torch.rand(images.shape[0], device=images.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=images.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(images.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = perturbation_x.view_as(y)
            D_yn = net(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = torch.randn_like(y) * sigma
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=128, N=3072, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D
        self.N = N
        self.gamma = gamma
        self.opts = opts
        print(f"In EDM loss: D:{self.D}, N:{self.N}")

    def __call__(self, net, images, labels=None, augment_pipe=None, stf=False, pfgmpp=False, ref_images=None):

        if pfgmpp:

            rnd_normal = torch.randn(images.shape[0], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            if self.opts.lsun:
                # use larger sigma for high-resolution datasets
                sigma *= 380. / 80

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=images.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(images.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))

            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = perturbation_x.view_as(y)
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        else:
            rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            if self.opts.lsun:
                # use larger sigma for high-resolution datasets
                sigma *= 380. / 80.
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = torch.randn_like(y) * sigma
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        if stf:
            ref_images[len(y):], augment_labels_2 = augment_pipe(ref_images[len(y):]) \
                if augment_pipe is not None else (images, None)
            # update augmented original images
            ref_images[:len(y)] = y
        if pfgmpp:
            if stf:
                target = self.pfgmpp_target(r.squeeze(), y+n, ref_images)
                target = target.view_as(y)
            else:
                target = y
        elif stf:
            # Diffusion (D-> \inf)
            target = self.stf_scores(sigma.squeeze(), y+n, ref_images)
            target = target.view_as(y)
        else:
            target = y

        loss = weight * ((D_yn - target) ** 2)
        return loss

    def stf_scores(self, sigmas, perturbed_samples, samples_full):

        with torch.no_grad():
            #print("perturbed shape:", perturbed_samples.shape, "full shape:", samples_full.shape)
            perturbed_samples_vec = perturbed_samples.reshape((len(perturbed_samples), -1))
            samples_full_vec = samples_full.reshape((len(samples_full), -1))

            gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - samples_full_vec) ** 2,
                                    dim=[-1])
            gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)
            distance = distance[:, :, None]
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))
            target = samples_full_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)

            gt_direction = torch.sum(weights * target, dim=1)

            return gt_direction

    def pfgm_target(self, perturbed_samples_vec, samples_full):
        real_samples_vec = torch.cat(
            (samples_full.reshape(len(samples_full), -1), torch.zeros((len(samples_full), 1)).to(samples_full.device)),
            dim=1)

        data_dim = self.N + self.D
        gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                                dim=[-1]).sqrt()

        # For numerical stability, timing each row by its minimum value
        distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
        distance = distance ** data_dim
        distance = distance[:, :, None]
        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
        coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
        diff = - (perturbed_samples_vec.unsqueeze(1) - real_samples_vec)

        # Calculate empirical Poisson field (N+D dimension in the augmented space)
        gt_direction = torch.sum(coeff * diff, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)

        # Normalizing the N+D-dimensional Poisson field
        gt_norm = gt_direction.norm(p=2, dim=1)
        gt_direction /= (gt_norm.view(-1, 1) + self.gamma)
        gt_direction *= np.sqrt(data_dim)

        target = gt_direction
        target[:, -1] = target[:, -1] / np.sqrt(self.D)

        return target

    def pfgmpp_target(self, r, perturbed_samples, samples_full):
        # # Augment the data with extra dimension z
        perturbed_samples_vec = torch.cat((perturbed_samples.reshape(len(perturbed_samples), -1),
                                           r[:, None]), dim=1).double()
        real_samples_vec = torch.cat(
            (samples_full.reshape(len(samples_full), -1), torch.zeros((len(samples_full), 1)).to(samples_full.device)),
            dim=1).double()

        data_dim = self.N + self.D
        gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                                dim=[-1]).sqrt()

        # For numerical stability, timing each row by its minimum value
        distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
        distance = distance ** data_dim
        distance = distance[:, :, None]
        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
        coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)

        target = real_samples_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)
        # Calculate empirical Poisson field (N+D dimension in the augmented space)
        gt_direction = torch.sum(coeff * target, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)
        gt_direction = gt_direction[:, :-1].float()

        return gt_direction

    def pfgm_perturation(self, samples, r):

        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                      size=samples.shape[0]).astype(np.double)
        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        inverse_beta = torch.from_numpy(inverse_beta).to(samples.device).double()
        # Sampling from p_r(R) by change-of-variable
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = samples_norm.view(len(samples_norm), -1)
        # Uniformly sample the angle direction
        gaussian = torch.randn(samples.shape[0], self.N).to(samples_norm.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * samples_norm
        perturbation_x = perturbation_x.float()

        return samples + perturbation_x.view_as(samples)
#----------------------------------------------------------------------------
