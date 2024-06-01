import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from core.cal import normalize
import os, random, sys
import scipy.io as sio

from scipy import sparse

class LandUse(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'LandUse-21.mat')
        train_x = []
        train_x.append(sparse.csr_matrix(data['X'][0, 0]).A.astype('float32'))  # (2100,20)
        train_x.append(sparse.csr_matrix(data['X'][0, 1]).A.astype('float32'))  # (2100,59)
        train_x.append(sparse.csr_matrix(data['X'][0, 2]).A.astype('float32'))  # (2100,40)
        index = random.sample(range(train_x[0].shape[0]), 2100) #相当于打乱顺序
        self.view1 = train_x[0][index]
        self.view2 = train_x[1][index]
        self.view3 = train_x[2][index]
        self.labels = np.squeeze(data['Y']).reshape(len(data['Y']),-1).astype('int')[index] # (2100,40)

    def __len__(self):
        return 2100

    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
