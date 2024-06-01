import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from core.cal import normalize
import os, random, sys
import scipy.io as sio

from scipy import sparse

class Scene(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'Scene-15.mat')
        X = data['X'][0]
        self.view1 = X[0].astype('float32')  #(4485,20)
        self.view2 = X[1].astype('float32')  #(4485,59)
        self.view3 = X[2].astype('float32')  #(4485,40)                               #(4485,40)
        #self.labels = np.squeeze(data['Y']).astype('int') #(4485,)
        self.labels = np.squeeze(data['Y']).reshape(len(np.squeeze(data['Y']).astype('int')), -1).astype('int')

    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(
            np.array(idx)).long()
