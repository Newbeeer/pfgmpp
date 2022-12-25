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
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
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
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=128, N=3072, gamma=5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D
        self.N = N
        self.gamma = gamma

    def __call__(self, net, images, labels=None, augment_pipe=None, stf=False, pfgm=False, pfgmv2=False, ref_images=None):
        if pfgm:
            r_min = 0.55 / np.sqrt(3072 / (self.D - 2 - 1))
            r_max = 2500 / np.sqrt(3072 / (self.D - 2 - 1))

            s = torch.rand(images.shape[0], device=images.device)
            # restrict #
            r_restrict = 230 / np.sqrt(
                self.N / (self.D - 2 - 1))
            s_restrict = (np.log(r_restrict) - np.log(r_min)) / \
                         (np.log(r_max) - np.log(r_min))
            s = torch.rand(images.shape[0], device=images.device)
            s[: int(len(images) * 0.6)] = torch.rand(
                int(len(images) * 0.6),
                device=images.device) * s_restrict

            r = r_min * (r_max / r_min) ** s

            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=(self.D - 1) / 2.,
                                          size=images.shape[0])
            inverse_beta = samples_norm / (1 - samples_norm)
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device)
            # Sampling from p_r(R) by change-of-variable
            samples_norm = torch.sqrt(r ** 2 * inverse_beta)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(images.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.view_as(images)

            # Perturb x
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            perturbed_x = y + perturbation_x
            net_x, net_z = net(perturbed_x, r, labels, self.D, augment_labels=augment_labels)
            net_x = net_x.view(net_x.shape[0], -1)
            # Predicted N+D-dimensional Poisson field
            D_yn = torch.cat([net_x, net_z[:, None]], dim=1)

            # # Augment the data with extra dimension z
            perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(images), -1),
                                               r[:, None]), dim=1).float()
            weight = torch.ones((len(perturbed_samples_vec), 1), device=images.device)
        elif pfgmv2:
            rnd_normal = torch.randn(images.shape[0], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()

            r = sigma * np.sqrt(self.D)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=images.shape[0])
            inverse_beta = samples_norm / (1 - samples_norm)
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device)
            # Sampling from p_r(R) by change-of-variable
            samples_norm = torch.sqrt(r ** 2 * inverse_beta)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(images.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = perturbation_x.view_as(y)
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        else:
            rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            sigma = torch.ones_like(sigma) * 5
            # sigma = torch.ones_like(sigma) * 0.01
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = torch.randn_like(y) * sigma
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        ref_images[len(y):], augment_labels_2 = augment_pipe(ref_images[len(y):]) \
            if augment_pipe is not None else (images, None)
        # update augmented original images
        ref_images[:len(y)] = y
        if stf:
            target = self.stf_scores(sigma.squeeze(), y+n, ref_images)
            target = target.view_as(y)
        elif pfgm:
            target = self.pfgm_target(perturbed_samples_vec, ref_images)
            target = target.view_as(D_yn)
            #print(target.norm(p=2, dim=1).mean())
        elif pfgmv2:
            target = self.pfgmv2_target(r.squeeze(), y+n, ref_images)
            print("target shape:", target.shape)
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

            # print(torch.sort(distance.squeeze(), dim=1, descending=True)[0])
            # print("idx:", torch.sort(distance.squeeze(), dim=1, descending=True)[1][:, 0])
            # diff = - (perturbed_samples_vec.unsqueeze(1) - samples_full_vec)
            # gt_direction2 = torch.sum(distance * diff, dim=1)
            target = samples_full_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)
            #print("stf weights:", torch.sort(weights.squeeze(), dim=1, descending=True)[0][:, 0])

            gt_direction = torch.sum(weights * target, dim=1)
            # perturbed_samples_vec = torch.cat((perturbed_samples_vec,
            #                                    torch.ones((len(perturbed_samples), 1)).to(perturbed_samples_vec.device) * sigmas.unsqueeze(1) * np.sqrt(self.D))
            #                                   , dim=1)
            #self.pfgm_target(perturbed_samples_vec, samples_full)
            #self.pfgmv2_target(sigmas * np.sqrt(self.D), perturbed_samples, samples_full)
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
        #print("pfgm weights:", torch.sort(coeff.squeeze(), dim=1, descending=True)[0][:, 0])
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

    def pfgmv2_target(self, r, perturbed_samples, samples_full):
        # # Augment the data with extra dimension z
        perturbed_samples_vec = torch.cat((perturbed_samples.reshape(len(perturbed_samples), -1),
                                           r[:, None]), dim=1).float()
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
        #print("pfgmv2 weights:", torch.sort(coeff.squeeze(), dim=1, descending=True)[0][:, 0])

        target = real_samples_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)
        # Calculate empirical Poisson field (N+D dimension in the augmented space)
        gt_direction = torch.sum(coeff * target, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)
        gt_direction = gt_direction[:, :-1]

        return gt_direction
#----------------------------------------------------------------------------
