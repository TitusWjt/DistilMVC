import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)  #(2500,1750)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)  #(2500,79)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()  #(2500,1)
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
