import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
import pandas as pd
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from sklearn import metrics


# helper functions to support computations
def gamma(t):
    return torch.exp(-t)


def beta(t):
    return torch.sqrt(1 - torch.exp(-2 * t))


def beta2(t):
    return 1 - torch.exp(-2 * t)


def compute_x(X_0, eps, delta_t):
    """
    Computes forward trajectories using the recursive formula: X_{n+1} = gamma_1 * X_n + beta_1 * eps_n
    :param X_0: Sample from initial distribution q0
    :param eps: Matrix sampled from a standard normal distribution. eps.shape = [num_observations, num_iterations, num_features]
    :param delta_t: Vector containing delta_t values
    :return: X_matrix: A matrix containing the trajectory of each observation across all iterations under the forward process
             X_matrix.shape = [num_observations, num_iterations, num_features]
    """
    X_matrix = torch.zeros(eps.shape)
    Xn = X_0
    X_matrix[:, 0, :] = X_0
    for n in range(1, X_matrix.size(dim=1)):
        Xn = gamma(delta_t[n]) * Xn + beta(delta_t[n]) * eps[:, n, :]
        X_matrix[:, n] = Xn

    return X_matrix


def compute_eps_with_negative_prompt(eps_pos, eps_neg, w1, w2):
    '''
    Computes a weighted sum of eps_theta from positive and negative prompts.
    :param eps_pos: eps_theta from the "positive prompt" (i.e., the distribution from which we want to sample)
    :param eps_neg: eps_theta from the "negative prompt" (i.e., the distribution we want to avoid)
    :param w1: Weight for eps_pos (i.e., how much you want to go towards eps_pos)
    :param w2: Weight for eps_neg (i.e., how much you want to avoid eps_neg)
    :return: eps_theta, the predicted value of eps_0
    '''
    # project eps_neg onto eps_pos
    proj = (torch.einsum('ij,ij->i', eps_neg, eps_pos) / torch.einsum('ij,ij->i', eps_pos, eps_pos)).unsqueeze(
        1) * eps_pos
    # get vector orthogonal to eps_pos
    eps_perp_neg = w2 * (eps_neg - proj)
    return w1 * eps_pos - eps_perp_neg


def compute_mu_with_eps(xnp1, tnp1, eps_theta, dt):
    '''
    Computes the approximated value of mu using eps_theta generated by the model.
    :param xnp1: Value of trajectory at time step n+1
    :param tnp1: Time value at time step n+1
    :param eps_theta: Predicted value of eps_0 generated by neural network
    :param dt: Time interval value at time step n+1
    :return: Mu, the predicted mean of Xn
    '''
    return (1 / gamma(dt)) * (xnp1 - (beta2(dt) / beta(tnp1)) * eps_theta)


def discretize_time(beta0, betaTn, num_iters):
    '''
    Uses a logarithmic discretization scheme to produce discrete time steps. Adjust parameters in main_network.py
    :param beta0: Initial value of beta_t
    :param betaTn: Final value of beta_t
    :param num_iters: Number of time steps
    :return: Two vectors: t and delta_t. The latter contains the time interval at each step and the former contains the
             time values (i.e., the partial sums of delta_t)
    '''
    variances = torch.linspace(beta0, betaTn, num_iters)
    delta_t = -0.5 * torch.log(1 - variances)
    return np.cumsum(delta_t), delta_t



def sample_cheatcode(cfg, data):
    '''
    "cheatcode" method to perform reverse diffusion without training.
    note: the data produced by this method will be heavily overfit to the original sample (i.e., the data generated
    by this method will look essentially identical to the original sample). Thus, data created with this method
    may not be desirable. One may use this method for experimentation and verifying results.
    :param cfg: Hyperparameter dictionary set in main_network.py
    :param data: Dictionary containing train, test, and validation sets
    :return: A matrix containing the trajectory of each observation across all iterations under the reverse process
    '''
    num_observations = cfg['num_observations_inference']
    num_iters = cfg['num_iters']
    num_features = data['num_features']

    t, _ = discretize_time(cfg['beta0'], cfg['betaTn'], num_iters)
    X0 = torch.from_numpy(data['X_train']).type(torch.float32)
    gamma_t = gamma(t)
    beta2_t = beta2(t)
    betas = torch.linspace(cfg['beta0'], cfg['betaTn'], num_iters)
    betas_sqrt = torch.sqrt(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)  # [exp(-2t_1) ... exp(-2t_1000)]
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # [1   ... exp(-2t_999)]
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    with torch.no_grad():
        # Sampling Algorithm
        X_matrix = torch.zeros((num_observations, num_iters, num_features))
        X_k = torch.randn((num_observations, num_features))
        for k in reversed(range(1, num_iters)):
            # reconstruction x_0
            s_k = -torch.cdist(X_k.type(torch.float32), gamma_t[k] * X0, p=2) ** 2 / (2 * beta2_t[k])
            a_k = F.softmax(s_k, dim=1)
            X0_hat = torch.matmul(a_k, X0)
            # update
            s1 = posterior_mean_coef1[k]
            s2 = posterior_mean_coef2[k]
            mu_theta = s1.reshape(-1, 1) * X0_hat + s2.reshape(-1, 1) * X_k
            z = torch.randn_like(X_k) if k > 1 else 0  # z ~ N(0,1)
            # Reverse Diffusion Process. Will run Until k=0. x_0 is the Generated Sample
            X_k = mu_theta + betas_sqrt[k] * z
            X_matrix[:, k - 1, :] = X_k

    return X_matrix.cpu().numpy()


def MMD(X, Y, bandwidth=1.0):
    '''
    Maximum Mean Discrepancy, which is used to compute distance between distributions given two samples.
    :param X: Sample 1
    :param Y: Sample 2
    :param bandwidth: Adjustable bandwidth parameter in the Gaussian kernel function used to compute MMD
    :return: A scalar representing the MMD between X and Y
    '''
    X = X.detach().numpy() if torch.is_tensor(X) else X
    Y = Y.detach().numpy() if torch.is_tensor(Y) else Y
    XX = metrics.pairwise.rbf_kernel(X, X, bandwidth)
    YY = metrics.pairwise.rbf_kernel(Y, Y, bandwidth)
    XY = metrics.pairwise.rbf_kernel(X, Y, bandwidth)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def MMD_max(X, Y, min_bandwidth=0.01, max_bandwidth=3, num_steps=15):
    '''
    Plots the Maximum Mean Discrepancy (MMD) between two samples over several bandwidths uniformly distributed
    between min_bandwidth and max_bandwidth
    X: Sample 1
    Y: Sample 2
    min_bandwidth: Minimum value for bandwidth
    max_bandwidth: Maximum value for bandwidth
    num_steps: Number of bandwidth values being evaluated
    '''
    X = X.detach().numpy() if torch.is_tensor(X) else X
    Y = Y.detach().numpy() if torch.is_tensor(Y) else Y
    bandwidth_values = np.linspace(min_bandwidth, max_bandwidth, num_steps)
    MMD_list = np.zeros(num_steps)
    for idx, bandwidth in enumerate(bandwidth_values):
        MMD_list[idx] = MMD(X, Y, bandwidth=bandwidth)
    return np.max(MMD_list)
