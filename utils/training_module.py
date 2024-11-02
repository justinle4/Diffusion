# this script is responsible for training a diffusion model
# to train a diffusion model, the forward process is simulated, and the model is optimized to predict the quantity eps_0

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from utils.models import *
from utils.toolbox import *
from utils.diffusion_formulas import *
from utils.inference_module import *


def trainModel(myModel, cfg, data):
    # unpack hyperparameters
    num_epochs = cfg['num_epochs']
    learning_rate = cfg['learning_rate']
    pretrained_weights = cfg['pretrained_weights']
    num_iters = cfg['num_iters']

    num_features = data['num_features']
    X_train = torch.from_numpy(data['X_train'].astype(np.float32))
    X_valid = torch.from_numpy(data['X_valid'].astype(np.float32))

    # model
    if pretrained_weights:
        myModel.load_state_dict(torch.load(pretrained_weights, map_location=torch.device('cpu'), weights_only=True))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
    df = pd.DataFrame(columns=('epoch', 'loss_train', 'loss_valid', 'mmd_train', 'mmd_valid'))

    # preliminary calculations
    t, delta_t = discretize_time(cfg['beta0'], cfg['betaTn'], num_iters)
    n_vector = torch.arange(1, num_iters + 1)

    # training loop
    t0 = time.time()
    X_train_batches = X_train.split(cfg['batch_size'])
    X_valid_batches = X_valid.split(cfg['batch_size'])

    for epoch in range(num_epochs):
        list_loss_train, list_loss_valid = [], []
        myModel.train()
        for X_0_train in X_train_batches:
            optimizer.zero_grad()

            # generate forward process
            num_observations_train = X_0_train.size(dim=0)
            n_repeated_train = n_vector.repeat(num_observations_train)
            eps_train = torch.randn(num_observations_train, num_iters, num_features)
            X_trajectories_train = compute_x(X_0_train, eps_train, delta_t)
            X_vectorized_train = X_trajectories_train.reshape(num_observations_train * num_iters, num_features)
            X_and_t_train = torch.cat([X_vectorized_train, n_repeated_train.unsqueeze(1)], dim=1)
            # compute true and predicted eps_0
            eps_pred_train = myModel(X_and_t_train)
            eps_0_train = (X_trajectories_train - X_0_train.unsqueeze(1) * gamma(t).unsqueeze(1)) / beta(t).unsqueeze(1)
            # loss
            loss_train = criterion(eps_0_train.reshape(num_observations_train * num_iters, num_features), eps_pred_train)
            list_loss_train.append(loss_train.item())

            # update parameters
            nn.utils.clip_grad_norm_(myModel.parameters(), 1.0)
            loss_train.backward()
            optimizer.step()

        # evaluate model on validation
        with torch.no_grad():
            for X_0_valid in X_valid_batches:
                num_observations_valid = X_0_valid.size(dim=0)
                n_repeated_valid = n_vector.repeat(num_observations_valid)
                eps_valid = torch.randn(num_observations_valid, num_iters, num_features)
                X_trajectories_valid = compute_x(X_0_valid, eps_valid, delta_t)
                X_vectorized_valid = X_trajectories_valid.reshape(num_observations_valid * num_iters, num_features)
                X_and_t_valid = torch.cat([X_vectorized_valid, n_repeated_valid.unsqueeze(1)], dim=1)
                eps_pred_valid = myModel(X_and_t_valid)
                eps_0_valid = (X_trajectories_valid - X_0_valid.unsqueeze(1) * gamma(t).unsqueeze(1)) / beta(t).unsqueeze(1)
                loss_valid = criterion(eps_0_valid.reshape(num_observations_valid * num_iters, num_features), eps_pred_valid)
                list_loss_valid.append(loss_valid.item())

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            X_0_generated = generate_sample(myModel, None, cfg, data, num_observations=cfg['num_observations_inference'])[:, 0, :]
            mmd_train = MMD_max(X_train, X_0_generated)
            mmd_valid = MMD_max(X_valid, X_0_generated)
        else:
            mmd_train = mmd_valid = float("nan")

        # save results
        mean_loss_train = np.mean(list_loss_train)
        mean_loss_valid = np.mean(list_loss_valid)
        df.loc[epoch] = [epoch, mean_loss_train, mean_loss_valid, mmd_train, mmd_valid]
        print(f"Epoch {epoch + 1} complete! Loss Train: {mean_loss_train}. MMD Train: {mmd_train}. Loss Valid: {mean_loss_valid}. MMD Valid: {mmd_valid}.")

    seconds_elapsed = time.time() - t0
    cfg['time_training'] = '{:.0f} minutes, {:.0f} seconds'.format(seconds_elapsed // 60, seconds_elapsed % 60)
    print(f"--- Training Complete in {cfg['time_training']}  ---")
    print(f"Trained {sum(p.numel() for p in myModel.parameters() if p.requires_grad)} parameters")

    # save model and training information
    save_network(myModel, cfg, df)
