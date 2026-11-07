'''
Description:
    Loss functions.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import geomloss


# ===========================================

def MSELoss(true_obs, est_obs):
    '''
    Mean squared error (MSE).
    :param true_obs (torch.FloatTensor): True expression data.
    :param est_obs (torch.FloatTensor): Predicted expression data.
    :return: (float) MSE value.
    '''
    loss_func = nn.MSELoss(reduction="mean")
    return loss_func(est_obs, true_obs)


def SinkhornLoss(true_obs, est_obs, blur=0.05, scaling=0.5, batch_size=None):
    '''
    Wasserstein distance computed by Sinkhorn algorithm.
    :param true_obs (torch.FloatTensor): True expression data.
    :param est_obs (torch.FloatTensor): Predicted expression data.
    :param blur (float): Sinkhorn algorithm hyperparameter. Default as 0.05.
    :param scaling (float): Sinkhorn algorithm hyperparameter. Default as 0.5.
    :param batch_size (None or int): Either None indicates using all true cell in computation, or an integer indicates
                                     using only a batch of true cells to save computational costs.
    :return: (float) Wasserstein distance average over all timepoints.
    '''
    n_tps = len(true_obs)
    loss = 0.0
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    for t in range(n_tps):
        t_est = est_obs[:, t, :]
        t_true = true_obs[t]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(t_true.shape[0]), size = batch_size, replace = (t_true.shape[0] < batch_size))
            t_true = t_true[cell_idx, :]
        t_loss = ot_solver(t_true, t_est)
        loss += t_loss
    loss = loss / n_tps
    return loss


# ===========================================
# Ablation loss functions (psoriasis project)
# ===========================================

def NegBinLoss(true_obs, mu, theta):
    '''
    Negative Binomial NLL as a replacement for MSE reconstruction loss.

    scRNA-seq data is count-based and overdispersed, making NB a better
    generative assumption than Gaussian/MSE. scVI uses the same parameterization.

    NB is parameterized by mean mu (decoder output) and dispersion theta
    (learned separately). The lgamma formulation handles continuous inputs
    (e.g. log-normalized expression) without requiring integer values.

    :param true_obs (torch.FloatTensor): True expression (cells x genes).
    :param mu (torch.FloatTensor): Predicted mean, same shape as true_obs.
    :param theta (torch.FloatTensor): Dispersion parameter, shape (n_genes,) or scalar.
    :return: (float) Mean NB NLL.
    '''
    eps = 1e-8
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)
    true_obs = true_obs.clamp(min=0)
    log_prob = (
        torch.lgamma(true_obs + theta)
        - torch.lgamma(theta)
        - torch.lgamma(true_obs + 1)
        + theta * torch.log(theta / (theta + mu))
        + true_obs * torch.log(mu / (theta + mu))
    )
    return -log_prob.mean()


def NonLesionalAlignmentLoss(latent_seq, nonlesional_centroid, last_tp_idx=-1):
    '''
    Auxiliary loss pulling predicted latents at the final time point toward
    the non-lesional day-0 centroid in latent space.

    Directly optimizes for RQ3: predicted post-treatment cells should move
    toward a healthier (non-lesional) state. The centroid is computed by
    encoding non-lesional day-0 cells through the current encoder at each
    training step, so it stays meaningful as the encoder learns.

    Non-lesional cells are never in the training set (held out per protocol),
    so this loss does not cause data leakage -- it only uses them as a
    reference direction, not as supervision targets.

    :param latent_seq (torch.FloatTensor): ODE latent trajectory,
        shape (batch_size, n_tps, latent_dim).
    :param nonlesional_centroid (torch.FloatTensor): Mean latent of
        non-lesional day-0 cells, shape (latent_dim,).
    :param last_tp_idx (int): Time point index to align (default: last tp).
    :return: (float) MSE between mean predicted latent and non-lesional centroid.
    '''
    pred_last = latent_seq[:, last_tp_idx, :]   # (batch_size, latent_dim)
    pred_mean = pred_last.mean(dim=0)            # (latent_dim,)
    return nn.MSELoss()(pred_mean, nonlesional_centroid)