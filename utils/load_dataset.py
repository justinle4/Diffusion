import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch.distributions as dist
import torch
from sklearn.preprocessing import StandardScaler
import random


def load_creditcard(cfg, fraud=True):
    '''
    Loads data from the creditcard dataset. Source for data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
    :param cfg: hyperparameter dictionary set in main_network.py
    :param fraud: controls whether to use the "fraud" or "legitimate" data. True for fraud, False for legitimate
    '''
    # load csv
    df_raw = pd.read_csv('datasets/creditcard.csv')
    # clean up df
    df = df_raw.copy()
    df.drop_duplicates(inplace=True)  # remove duplicate
    df.drop(['Time'], axis=1, inplace=True)  # drop time

    # split
    df_train, df_test = train_test_split(df, test_size=cfg['prop_test'], stratify=df['Class'],
                                         random_state=cfg['seed_nbr'])

    if fraud:  # take fraud transactions
        df_train = df_train[df_train['Class'] == 1]
        df_test = df_test[df_test['Class'] == 1]
        dataset_name = 'creditcard_fraud'
    else:  # take legitimate transactions
        df_train = df_train[df_train['Class'] == 0]
        df_test = df_test[df_test['Class'] == 0]
        df_train = df_train.sample(cfg['num_observations_training'], random_state=cfg['seed_nbr'])  # legitimate data is too large -- take a sample
        dataset_name = 'creditcard_legit'

    # save mean and variance
    amount_mean = df_train['Amount'].mean()
    amount_std = df_train['Amount'].std()

    # Standardize variable 'Amount' to have zero mean and unit variance
    sc = StandardScaler()
    df_train['Amount'] = sc.fit_transform(df_train['Amount'].values.reshape(-1, 1))

    # A.6) numpy
    feature_names = df_train.drop('Class', axis=1).columns
    X_train = df_train.drop('Class', axis=1).values
    X_test = df_test.drop('Class', axis=1).values

    return X_train, X_test, feature_names, amount_mean, amount_std, dataset_name


def load_creditcard_cluster(cfg, filename):
    df = pd.read_csv(filename)
    feature_names = df.columns
    X_train = df.values
    dataset_name = 'cluster_3'
    return X_train, feature_names, dataset_name


def load_datasaurus(cfg):
    df_all = pd.read_csv("datasets/datasaurus.csv")
    df_dino = df_all[df_all["dataset"] == "dino"]
    feature_names = df_dino.drop('dataset', axis=1).columns

    # select randomly datapoint
    rng = np.random.default_rng(cfg['seed_nbr'])
    idx = rng.integers(0, len(df_dino), cfg['num_observations_training'])
    x_original = df_dino["x"].iloc[idx].tolist()
    y_original = df_dino["y"].iloc[idx].tolist()
    x = np.array(x_original) + .15 * rng.normal(size=len(x_original))
    y = np.array(y_original) + .15 * rng.normal(size=len(y_original))
    # rescale
    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)

    return X, feature_names, 'datasaurus'


def load_moons(cfg):
    X_train = datasets.make_moons(n_samples=cfg['num_observations_training'], shuffle=True, random_state=cfg['seed_nbr'])[0]
    return X_train, ['x', 'y'], 'moons'


def load_top_moon(cfg):
    X_train_raw = datasets.make_moons(n_samples=cfg['num_observations_training'], shuffle=True, random_state=cfg['seed_nbr'])
    X_train = X_train_raw[0][X_train_raw[1] == 0]
    return X_train, ['x', 'y'], 'top_moon'


def load_bottom_moon(cfg):
    X_train_raw = datasets.make_moons(n_samples=cfg['num_observations_training'], shuffle=True, random_state=cfg['seed_nbr'])
    X_train = X_train_raw[0][X_train_raw[1] == 1]
    return X_train, ['x', 'y'], 'bottom_moon'


def load_one_dirac(cfg, peak):
    '''
    Loads a dataset sampled from a dirac
    :param cfg: hyperparameter dictionary set in main_network.py
    :param peak: the location for the dirac
    '''
    X_train = peak + np.zeros((cfg['num_observations_training'], 1))
    return X_train, ['x'], 'one_dirac'


def load_two_diracs(cfg, peak1, peak2):
    '''
    Loads a dataset sampled from two equally weighted dirac masses
    :param cfg: hyperparameter dictionary set in main_network.py
    :param peak1: the location for the first dirac
    :param peak2: the location for the second dirac
    '''
    dirac1 = peak1 + np.zeros((cfg['num_observations_training'] // 2, 1))
    dirac2 = peak2 + np.zeros((cfg['num_observations_training'] // 2, 1))
    X_train = np.concatenate([dirac1, dirac2])
    return X_train, ['x'], 'two_diracs'


def load_uniform(cfg, a, b):
    '''
    Loads a dataset sampled from a uniform distribution
    :param cfg: hyperparameter dictionary set in main_network.py
    :param a: lower bound for the uniform
    :param b: upper bound for the uniform
    '''
    torch.manual_seed(cfg['seed_nbr'])
    q0 = dist.Uniform(a, b)
    X_train = q0.sample((cfg['num_observations_training'], 1)).numpy()
    return X_train, ['x'], 'uniform'


def load_dirac_and_uniform(cfg, peak, a, b):
    '''
    Loads a dataset that combines a uniform distribution with a dirac, each equally weighted
    :param cfg: hyperparameter dictionary set in main_network.py
    :param peak: the location for the dirac
    :param a: lower bound for the uniform
    :param b: upper bound for the uniform
    '''
    torch.manual_seed(cfg['seed_nbr'])
    q0 = dist.Uniform(a, b)
    uniform = q0.sample((cfg['num_observations_training'] // 2, 1)).numpy()
    dirac = peak + np.zeros((cfg['num_observations_training'] // 2, 1))
    X_train = np.concatenate([uniform, dirac])
    return X_train, ['x'], 'dirac_and_uniform'


def load_2D_uniform_grid(cfg):
    X = np.zeros((5, 5, 2))
    for i in range(0, 5):
        for j in range(0, 5):
            X[i, j, :] = [i - 2, j - 2]

    X_train = X.reshape(-1, 2).repeat(cfg['num_observations_training'] // 25, axis=0)
    return X_train, ['x', 'y'], '2D_uniform_grid'




