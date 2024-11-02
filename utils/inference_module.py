# this script uses diffusion models to recover an initial distribution from data sampled from a standard normal
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
from utils.vector_fields import *
from utils.load_dataset import *


def generate_sample(myModel, myModel_neg, cfg, data, num_observations=None):
    # unpack hyperparameters
    if not num_observations:  # default: create number of points as specified in cfg.
        num_observations = cfg['num_observations_inference']
    num_features = data['num_features']
    num_iters = cfg['num_iters']

    t, delta_t = discretize_time(cfg['beta0'], cfg['betaTn'], num_iters)
    X_final = torch.randn((num_observations, num_features))  # begin with points from a standard normal at t = T
    X_matrix = torch.zeros(num_observations, num_iters, num_features)
    X_matrix[:, -1, :] = X_final
    Xn = X_final

    # set seed for reproducibility
    torch.manual_seed(cfg['seed_nbr'])
    np.random.seed(cfg['seed_nbr'])

    for n in range(num_iters - 1, 0, -1):
        # take current time and combine x and t
        tn = t[n]
        dt = delta_t[n]

        X_and_t = torch.cat([Xn, n * torch.ones((num_observations, 1))], dim=1)

        # predict mu
        if myModel_neg:
            eps_pos = myModel(X_and_t).reshape(num_observations, num_features)
            eps_neg = myModel_neg(X_and_t).reshape(num_observations, num_features)
            eps_pred = compute_eps_with_negative_prompt(eps_pos, eps_neg, cfg['w1'], cfg['w2'])
        else:
            eps_pred = myModel(X_and_t)

        mu_hat = compute_mu_with_eps(Xn, tn, eps_pred, dt)

        # compute standard deviation
        sigma_n = beta(dt) * beta(tn) / beta(tn + dt)

        # generate values from standard normal
        eps = torch.randn(num_observations, num_features)

        # compute current xn
        Xn = mu_hat + sigma_n * eps

        # add xn to matrix
        X_matrix[:, n - 1, :] = Xn

    return X_matrix


def inference(myModel, cfg, data):
    # set seed for reproducibility
    torch.manual_seed(cfg['seed_nbr'])
    np.random.seed(cfg['seed_nbr'])

    X_train = data['X_train']
    if cfg['pretrained_weights'] and not cfg['training']:
        myModel.load_state_dict(torch.load(cfg['pretrained_weights'], map_location=torch.device('cpu'), weights_only=True))

    if cfg['negative_model']:
        if cfg['negative_model_weights']:
            myModel_neg = eval(cfg['negative_model'])()
            myModel_neg.load_state_dict(torch.load(cfg['negative_model_weights'], map_location=torch.device('cpu'), weights_only=True))
        else:
            print("Please provide negative model weights or set cfg['negative_model'] to False.")
            exit(-1)
    else:
        myModel_neg = None

    X_matrix = generate_sample(myModel, myModel_neg, cfg, data)
    X0 = X_matrix[:, 0, :].cpu().detach().numpy()

# use cheat code:
    X_matrix_cheat = sample_cheatcode(cfg, data)
    X0_cheat = X_matrix_cheat[:, 0, :]

    # credit card fraud: save as CSV
    if cfg['dataset_name'] in ['creditcard_fraud', 'creditcard_legit', 'cluster_1', 'cluster_2', 'cluster_3']:
        synthetic_data = pd.DataFrame(X0, columns=data['feature_names'])
        synthetic_data_cheat = pd.DataFrame(X0_cheat, columns=data['feature_names'])
        # rescale amount column
        synthetic_data['Amount'] = synthetic_data['Amount'] * data['amount_std'] + data['amount_mean']
        synthetic_data_cheat['Amount'] = synthetic_data_cheat['Amount'] * data['amount_std'] + data['amount_mean']
        # save synthetic data
        synthetic_data.to_csv(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/synthetic_data.csv', index=False)
        synthetic_data_cheat.to_csv(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/synthetic_data_cheat.csv', index=False)

    # 2D diffusion: scatterplot and vector fields
    elif data['num_features'] == 2:
        if cfg['negative_model']:
            plot_scatter_2D(X0, X_train, cfg, data, negative_prompt=True)
            plot_scatter_2D(X0_cheat, X_train, cfg, data, filename='synthetic_scatter_cheat.pdf', negative_prompt=True)
            plot_vector_fields_2D(myModel, X_train, cfg, myModel_neg=myModel_neg)
        else:
            plot_scatter_2D(X0, X_train, cfg, data)
            plot_scatter_2D(X0_cheat, X_train, cfg, data, filename='synthetic_scatter_cheat.pdf')
            plot_vector_fields_2D(myModel, X_train, cfg)

        plot_scatter_2D_timeline(X_matrix, cfg)
        plot_scatter_2D_timeline(X_matrix_cheat, cfg, filename='scatterplot_timeline_cheat.pdf')

    # 1D diffusion: plot trajectories, estimated q0, and vector fields
    elif data['num_features'] == 1:
        plot_trajectories_1D(X_matrix, cfg)
        plot_trajectories_1D(X_matrix_cheat, cfg, filename='reverse_trajectories_cheat.pdf')
        plot_q0_1D(X0.squeeze(1), cfg)
        plot_q0_1D(X0_cheat.squeeze(1), cfg, filename='estimated_q0_cheat.pdf')
        if cfg['negative_model']:
            plot_vector_fields_1D(myModel, X0, cfg, myModel_neg)
        else:
            plot_vector_fields_1D(myModel, X0, cfg)

    print(f'MMD on train set: {MMD_max(X_train, X0)}')
    print(f'MMD on train set (cheatcode): {MMD_max(X_train, X0_cheat)}')
    plot_MMD(X_train, X0, cfg, data)
    plot_MMD(X_train, X0_cheat, cfg, data, filename='MMD_plot_cheat.pdf')

    if 'X_test' in data:
        X_test = data['X_test']
        plot_MMD(X_test, X0, cfg, data)
        plot_MMD(X_test, X0_cheat, cfg, data, filename='MMD_plot_cheat.pdf')
        mmd_test = MMD_max(X_test, synthetic_data.values)
        mmd_test_cheat = MMD_max(X_test, X0_cheat)
        print(f"MMD on test set: {mmd_test}")
        print(f"MMD on test set (cheatcode): {mmd_test_cheat}")

    if cfg['negative_model']:
        print(f'MMD to Negative Prompt set: {MMD_max(data["negative_prompt"], X0)}')



