# this script plots vector fields that show the "flow" of the reverse process. Only 1D and 2D have been implemented.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from sklearn import datasets

from utils.models import *
from utils.toolbox import *
from utils.diffusion_formulas import *


def plot_vector_fields_2D(myModel, original_data, cfg, num_snapshots=10, myModel_neg=None):
    num_iters = cfg['num_iters']
    t, delta_t = discretize_time(cfg['beta0'], cfg['betaTn'], num_iters)
    x_min = np.min(original_data[:, 0])
    x_max = np.max(original_data[:, 0])
    y_min = np.min(original_data[:, 1])
    y_max = np.max(original_data[:, 1])

    x_range = np.linspace(x_min - 1, x_max + 1, 25)
    y_range = np.linspace(y_min - 1, y_max + 1, 25)
    # define snapshot iterations
    a = (num_iters-1) / (1.5 ** (num_snapshots - 1))
    iter_snapshots = [int(a*1.5**n) for n in range(num_snapshots)]
    iter_snapshots[0] = 0
    plt.figure(3)
    plt.clf()
    fig1, ax1 = plt.subplots(2, num_snapshots // 2, figsize=(20, 8))
    ax1 = ax1.flatten()

    # one vector field for each t
    with torch.no_grad():
        for idx_n, n in enumerate(iter_snapshots):
            current_idx = num_snapshots - idx_n - 1
            tn = t[n]
            dt = delta_t[n]
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            grid_u = np.zeros((len(x_range), len(y_range)))
            grid_v = np.zeros((len(x_range), len(y_range)))

            for idx_x, x_ in enumerate(x_range):
                for idx_y, y_ in enumerate(y_range):
                    X_and_t = torch.tensor(([x_, y_, n]), dtype=torch.float).unsqueeze(0)
                    if cfg['negative_model']:
                        eps_pos = myModel(X_and_t)
                        eps_neg = myModel_neg(X_and_t)
                        eps_pred = compute_eps_with_negative_prompt(eps_pos, eps_neg, cfg['w1'], cfg['w2'])
                    else:
                        eps_pred = myModel(X_and_t).squeeze(1)
                    mu_hat = compute_mu_with_eps(torch.tensor([x_, y_]), tn, eps_pred, dt).squeeze(0)
                    approx_dx = (mu_hat.numpy()[0] - x_) / dt.numpy()
                    approx_dy = (mu_hat.numpy()[1] - y_) / dt.numpy()
                    grid_u[idx_x, idx_y] = approx_dx
                    grid_v[idx_x, idx_y] = approx_dy

            ax1[current_idx].quiver(grid_x, grid_y, grid_u.T, grid_v.T, np.sqrt(grid_u**2 + grid_v**2), angles='xy', scale_units='xy', pivot='middle')
            ax1[current_idx].set_xlabel("X")
            ax1[current_idx].set_ylabel("Y")
            ax1[current_idx].scatter(original_data[:, 0], original_data[:, 1], label='Original Data', alpha=0.25)
            ax1[current_idx].legend()
            ax1[current_idx].set_title(f't = {tn:.5f}', fontsize=20)
            ax1[current_idx].grid()

    plt.tight_layout()
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + f'/2D_vector_field_timeline.pdf')
    plt.show()


def plot_vector_fields_1D(myModel, X0, cfg, myModel_neg=None):

    # compute time and space coordinates
    t, delta_t = discretize_time(cfg['beta0'], cfg['betaTn'], cfg['num_iters'])
    t_range = t[4::5]  # only keep every 5th time step to make vector fields clearer
    delta_t = delta_t[4::5]
    x_min = np.min(X0[:, 0])
    x_max = np.max(X0[:, 0])
    # x_range = np.linspace(x_min - 2, x_max + 2, 25)
    x_range = np.linspace(-12, 6, 40)
    grid_t, grid_x = np.meshgrid(t_range, x_range)
    grid_v = np.zeros((len(x_range), len(t_range)))

    with torch.no_grad():
        for idx_x, x_ in enumerate(x_range):
            for idx_t, t_ in enumerate(t_range):
                X_and_t = torch.tensor(([x_, 5 * idx_t + 4]), dtype=torch.float).unsqueeze(0)
                if cfg['negative_model']:
                    eps_pos = myModel(X_and_t)
                    eps_neg = myModel_neg(X_and_t)
                    eps_pred = compute_eps_with_negative_prompt(eps_pos, eps_neg, cfg['w1'], cfg['w2'])
                else:
                    eps_pred = myModel(X_and_t).squeeze(1)
                mu_hat = compute_mu_with_eps(torch.tensor([x_]), torch.tensor([t_]), eps_pred, delta_t[idx_t])
                approx_dx = (x_ - mu_hat.numpy()[0]) / delta_t[idx_t].numpy()
                grid_v[idx_x, idx_t] = approx_dx

    plt.figure(1)
    plt.clf()
    plt.quiver(grid_t, grid_x, -2 + grid_t * 0, -grid_v, -grid_v, scale_units='xy', pivot='middle', scale=100)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.title(f'Reverse Vector Fields')
    plt.grid()
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + "/1D_vector_field.pdf")
    plt.show()
