import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.load_dataset import *
from utils.models import *
from utils.training_module import *
from utils.inference_module import *

# pretrained models contained in the "models" folder:
# One Dirac (3): 'models/one_dirac.pth'
# Two Diracs (-1 and 1): 'models/two_diracs.pth'
# Uniform (3 to 5): 'models/uniform.pth'
# Two Moons: 'models/two_moons.pth'
# Top Moon: 'models/top_moon.pth'
# Bottom Moon: 'models/top_moon.pth'
# Datasaurus: 'models/datasaurus.pth'
# Creditcard (fraud): 'models/creditcard_fraud.pth'
# Creditcard (legit): 'models/creditcard_legit.pth'
# Cluster 1, 2, 3 (fraud): 'models/cluster_1.pth'

# note: credit card models were trained with 800 time steps. It is recommended to set num_iters to 800 when
#       using any pretrained credit card model

# declaring hyper parameters
cfg = {
    'dataset': 'load_creditcard',  # type in dataset function as it appears in utils/load_dataset.py
    'dataset_params': [True],  # express parameters for dataset function as a list (exclude cfg from parameters)
    'training': False,  # set as true to train the model and false to only run inference
    'seed_nbr': 42,
    'prop_test': .2,
    'prop_valid': '',  # set as an empty string or false if you do not want to split into a validation set.
    'num_observations_training': 1000,
    'batch_size': 64,
    'num_observations_inference': 1000,
    'beta0': 10 ** -4,
    'betaTn': 2 * 10 ** -2,
    'num_iters': 800,
    'num_epochs': 500,
    'learning_rate': 0.001,
    'model': 'Model3layerPosEnc_CC',  # see utils/models.py for a list of available architectures.
    'pretrained_weights': '',
    'negative_model': '',
    'negative_model_weights': '',
    'w1': 1,  # weight for positive prompt
    'w2': 0   # weight for negative prompt
}

data = {}

# load dataset
if cfg['dataset'] == 'load_creditcard':
    X_train, data['X_test'], data['feature_names'], data['amount_mean'], data['amount_std'], cfg['dataset_name'] = load_creditcard(cfg, fraud=cfg['dataset_params'][0])

elif cfg['dataset'] == 'load_creditcard_cluster':
    X_train, data['feature_names'], cfg['dataset_name'] = load_creditcard_cluster(cfg, cfg['dataset_params'][0])

elif cfg['dataset'] == 'load_moons':
    X_train, data['feature_names'], cfg['dataset_name'] = load_moons(cfg)

elif cfg['dataset'] == 'load_datasaurus':
    X_train, data['feature_names'], cfg['dataset_name'] = load_datasaurus(cfg)

elif cfg['dataset'] == 'load_top_moon':
    X_train, data['feature_names'], cfg['dataset_name'] = load_top_moon(cfg)

elif cfg['dataset'] == 'load_bottom_moon':
    X_train, data['feature_names'], cfg['dataset_name'] = load_bottom_moon(cfg)

elif cfg['dataset'] == 'load_one_dirac':
    X_train, data['feature_names'], cfg['dataset_name'] = load_one_dirac(cfg, peak=cfg['dataset_params'][0])

elif cfg['dataset'] == 'load_two_diracs':
    X_train, data['feature_names'], cfg['dataset_name'] = load_two_diracs(cfg, peak1=cfg['dataset_params'][0], peak2=cfg['dataset_params'][1])

elif cfg['dataset'] == 'load_uniform':
    X_train, data['feature_names'], cfg['dataset_name'] = load_uniform(cfg, a=cfg['dataset_params'][0], b=cfg['dataset_params'][1])

elif cfg['dataset'] == 'load_dirac_and_uniform':
    X_train, data['feature_names'], cfg['dataset_name'] = load_dirac_and_uniform(cfg, peak=cfg['dataset_params'][0], a=cfg['dataset_params'][1], b=cfg['dataset_params'][2])

elif cfg['dataset'] == 'load_2D_uniform_grid':
    X_train, data['feature_names'], cfg['dataset_name'] = load_2D_uniform_grid(cfg)

else:
    print("Dataset name not recognized. See utils/load_dataset.py for a list of datasets.")
    exit(-1)


if cfg['negative_model']:  # load the negative prompt dataset (for plotting/MMD purposes only)
    _, data['negative_prompt'], _, _, _, _ = load_creditcard(cfg, fraud=not cfg['dataset_params'][0])
    # data['negative_prompt'], _, _ = load_bottom_moon(cfg)
else:
    data['negative_prompt'] = False

data['num_features'] = len(data['feature_names'])

# bookkeeping
cfg['folder_result'] = 'results_ML/' + cfg['dataset_name']
cfg['str_time'] = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)
os.makedirs(cfg['folder_result'] + '/Report_' + cfg['str_time'])

# further split X_train into training and validation sets
if cfg['prop_valid']:
    if cfg['prop_valid'] + cfg['prop_test'] >= 1:
        print("Error: prop_valid and prop_test must sum to less than 1.")
        exit(-1)
    else:
        data['X_train'], data['X_valid'] = train_test_split(X_train, test_size=cfg['prop_valid'] / (1-cfg['prop_test']), random_state=cfg['seed_nbr'])
else:
    data['X_train'] = data['X_valid'] = X_train

# train/inference
myModel = eval(cfg['model'])()
if cfg['training']:
    trainModel(myModel, cfg, data)

inference(myModel, cfg, data)

print("--- Main network has fully completed running ---")
