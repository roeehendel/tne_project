import os
import random

import numpy as np
import torch


def initialize(random_seed: int):
    set_random_seed(random_seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
