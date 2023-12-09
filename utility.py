import pandas as pd
import os
import scanpy as sc
import numpy as np
import cv2
import torchvision.transforms as transforms
import sklearn.neighbors
import torch
import os
from torch_geometric.data import Data
import random
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False