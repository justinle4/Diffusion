import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
import os, json
import datetime
from utils.diffusion_formulas import *


def plot_stat_training(df, folder):
    '''
    Statistics over epochs
    df: Statistics dataframe with loss
    folder: Name of folder to save statistics
    '''
    # init
    nbEpochs = len(df) - 1
    # plot
    plt.figure(1);
    plt.clf()
    plt.ioff()
    plt.plot(df['epoch'], df['loss_train'], '-o', label='loss train')
    plt.plot(df['epoch'], df['loss_valid'], '-o', label='loss valid')
    mmd_indices = df['mmd_train'].notnull()
    plt.plot(df['epoch'][mmd_indices], df['mmd_train'][mmd_indices], '-o', label='mmd train')
    plt.plot(df['epoch'][mmd_indices], df['mmd_valid'][mmd_indices], '-o', label='mmd valid')
    plt.grid(visible=True, which='major')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend(loc=0)
    plt.draw()
    plt.savefig(folder + '/stat_epochs.pdf')
    plt.show()
    plt.close()


def save_network(myModel, cfg, df):
    '''
    Saves the trained model, hyperparameters, and training stats
    myModel: Model used for training
    cfg: Hyperparameter dictionary set in main_network.py
    df: Statistics dataframe with loss
    '''
    # save network
    torch.save(myModel.state_dict(), cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/model.pth')
    # save parameters
    with open(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/parameters.json', 'w') as jsonFile:
        json.dump(cfg, jsonFile, indent=2)
    # save stat training (and a plot)
    df.to_csv(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/stat_epochs.csv', index=False)
    plot_stat_training(df, cfg['folder_result'] + '/Report_' + cfg['str_time'])


def plot_trajectories_1D(X, cfg, filename='reverse_trajectories.pdf'):
    '''
    Plots trajectories computed from reverse diffusion
    X: Matrix containing all observations across all trajectories
    t: Time vector
    cfg: Hyperparameter dictionary set in main_network.py
    '''
    x_ = X.detach().numpy() if torch.is_tensor(X) else X
    t, _ = discretize_time(cfg['beta0'], cfg['betaTn'], cfg['num_iters'])
    t_ = t.detach().numpy()
    for n in range(x_.shape[0]):
        plt.plot(t_, x_[n])
    plt.xlabel('time')
    plt.title(f'Trajectories Under Reverse Diffusion')
    plt.grid()
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + "/" + filename)
    plt.show()


def plot_q0_1D(X, cfg, filename='estimated_q0.pdf'):
    '''
    Plots predicted initial distribution q0 obtained from reverse diffusion
    X: Vector containing the predicted initial position of all trajectories
    cfg: Hyperparameter dictionary set in main_network.py
    '''
    x_ = X.detach().numpy() if torch.is_tensor(X) else X
    x_min = np.min(x_)
    x_max = np.max(x_)
    kernel = scipy.stats.gaussian_kde(x_)  # bandwidth: kernel.factor
    int_x = np.linspace(x_min - 3, x_max + 3, 1000)
    q0_estimated = kernel.pdf(int_x)
    plt.figure(3)
    plt.clf()
    plt.plot(x_, 0 * x_, 'o', alpha=.3)
    plt.plot(int_x, q0_estimated)
    plt.xlabel('x')
    plt.title(f'Recovered distribution of X0')
    plt.grid()
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + "/" + filename)
    plt.show()


def plot_scatter_2D(synthetic_data, original_data, cfg, data, filename='synthetic_scatter.pdf', negative_prompt=False):
    '''
    Creates a scatterplot containing the synthetic data alongside the original training data
    synthetic_data: Matrix containing the x and y coordinates of each synthetic datapoint
    original_data: Matrix containing the x and y coordinates of each datapoint from the training set
    cfg: Hyperparameter dictionary set in main_network.py
    filename: Adjust filename of figure
    negative_prompt: Decides whether to include the negative prompt dataset in the final scatterplot
    '''
    synthetic_data = synthetic_data.detach().numpy() if torch.is_tensor(synthetic_data) else synthetic_data
    original_data = original_data.detach().numpy() if torch.is_tensor(original_data) else original_data
    plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], label='Synthetic Data')
    plt.scatter(original_data[:, 0], original_data[:, 1], label='Original Data', alpha=0.15)
    if negative_prompt:
        plt.scatter(data['negative_prompt'][:, 0], data['negative_prompt'][:, 1], label='Negative Prompt', alpha=0.15)
    leg = plt.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title('Recovered distribution of X0')
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/' + filename)
    plt.show()


def plot_scatter_2D_timeline(synthetic_data, cfg, num_snapshots=10, filename='scatterplot_timeline.pdf'):
    '''
    Creates a figure containing several time snapshots of the reverse diffusion process in 2D
    synthetic_data: Matrix containing all the synthetic data across all iterations
    cfg: Hyperparameter dictionary set in main_network.py
    num_snapshots: Number of desired snapshots in final figure
    filename: Adjust filename of figure
    '''
    plt.figure(2)
    num_iters = cfg['num_iters']
    t, _ = discretize_time(cfg['beta0'], cfg['betaTn'], num_iters)
    synthetic_data = synthetic_data.detach().numpy() if torch.is_tensor(synthetic_data) else synthetic_data
    fig1, ax1 = plt.subplots(2, num_snapshots // 2, figsize=(20, 8))
    ax1 = ax1.flatten()
    a = (num_iters - 1) / (1.5 ** (num_snapshots - 1))
    iter_snapshots = [int(a * 1.5 ** n) for n in range(num_snapshots)]
    iter_snapshots[0] = 0
    for idx_n, n in enumerate(iter_snapshots):
        current_idx = num_snapshots - idx_n - 1
        ax1[current_idx].scatter(synthetic_data[:, n, 0], synthetic_data[:, n, 1])
        if idx_n == 0:
            ax1[current_idx].set_title(f't = 0', fontsize=20)
        else:
            ax1[current_idx].set_title(f't = {t[n]:.5f}', fontsize=20)
        ax1[current_idx].tick_params(left=False,
                                     bottom=False,
                                     labelleft=False,
                                     labelbottom=False)
    plt.tight_layout()
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/' + filename)
    plt.show()


def plot_MMD(X, Y, cfg, data, min_bandwidth=0.05, max_bandwidth=3, num_steps=50, filename='MMD_plot.pdf'):
    '''
    Plots the Maximum Mean Discrepancy (MMD) between two samples over several bandwidths uniformly distributed
    between min_bandwidth and max_bandwidth
    X: Sample 1
    Y: Sample 2
    min_bandwidth: Minimum value for bandwidth
    max_bandwidth: Maximum value for bandwidth
    num_steps: Number of bandwidth values being plotted
    filename: Adjust filename of figure
    '''
    bandwidth_values = np.linspace(min_bandwidth, max_bandwidth, num_steps)
    MMD_list = np.zeros(num_steps)
    for idx, bandwidth in enumerate(bandwidth_values):
        MMD_list[idx] = MMD(X, Y, bandwidth=bandwidth)
    plt.plot(bandwidth_values, MMD_list, label='MMD of Synthetic and Test Data')
    # plot MMD of test and train data if a test set is used
    if 'X_test' in data:
        X_test = data['X_test']
        X_train = data['X_train']
        bandwidth_values = np.linspace(min_bandwidth, max_bandwidth, num_steps)
        self_MMD_list = np.zeros(num_steps)
        for idx, bandwidth in enumerate(bandwidth_values):
            self_MMD_list[idx] = MMD(X_train, X_test, bandwidth=bandwidth)
        plt.plot(bandwidth_values, self_MMD_list, label='MMD of Train and Test Data')
    plt.xlabel("Bandwidth")
    plt.ylabel("MMD")
    plt.title("MMD Values for Varying Bandwidths")
    plt.legend()
    plt.savefig(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/' + filename)
    plt.show()
