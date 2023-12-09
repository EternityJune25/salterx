import os
import numpy as np
import pandas as pd
import h5py
import time
from scipy.sparse import csr_matrix
import anndata
import torch
from torch.utils.data import Dataset
class GraphDS(Dataset):
    def __init__(self, name):
        super(GraphDS,self).__init__()
        root='./'
        tic = time.time()
        rna = anndata.read_h5ad(root+'data/pbmc10k/PBMC_babel_RNA_rawcount.h5ad')
        atac = anndata.read_h5ad(root+'data/pbmc10k/PBMC_babel_ATAC_rawcount.h5ad')
        self.rna_data=rna.X
        print(rna.X)
        self.atac_data=atac.X
        tok = time.time()
        print("Finish loading in {} s, data size {} and {}".format(round(tok-tic, 2),self.rna_data.shape,self.atac_data.shape))
GraphDS('大乌龟')