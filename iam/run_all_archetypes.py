import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
import time, os, sys, pickle
import numpy as np
from scripts.utils import *
from scripts.ds_class import *
from scripts.homeos import *
from scripts.plotting import *
from scripts.fit_motif import *
from scripts.time_series import *
from scripts.ra import *
from scripts.exp_tools import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns

set_seed(313)
from pathlib import Path
exp_dir = Path('experiments')
data_dir = exp_dir / 'all_targets'


npy_files = list(data_dir.glob('*.npy'))
print(data_dir)
for file in npy_files:
    target_name = file.name.removesuffix('.npy')

    target_trajs = np.load(file)
    print(target_name, 'dim = ', target_trajs.shape[2])
    file_ = data_dir / f'{target_name}.npy'

archetypes_2d = ['lds', 'lc', 'ring', 'bla', 'bistable']


archetypes_2d = ['bistable']

save_dir = data_dir / 'motif_fits' 
save_dir.mkdir(parents=True, exist_ok=True)

for file in npy_files:
        target_name = file.name.removesuffix('.npy')
        print("Starting training archetype on system", target_name)
        for archetype in archetypes_2d:
                print('training archetype', archetype)
                run_on_target(target_name, save_dir=save_dir, data_dir=data_dir, ds_motif=archetype, analytic=True, canonical=True, jac_lambda_reg=.0, two_phase=False)