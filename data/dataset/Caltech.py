import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from core.cal import normalize


class Caltech(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'Caltech101-20.mat')
        X = data['X'][0]
        self.view1 = normalize(X[0]).astype('float32') #(2386,48)
        self.view2 = normalize(X[1]).astype('float32') #(2386,40)
        self.view3 = normalize(X[2]).astype('float32') #(2386,254)
        self.view4 = normalize(X[3]).astype('float32') #(2386,1984)
        self.view5 = normalize(X[4]).astype('float32') #(2386,512)
        self.view6 = normalize(X[5]).astype('float32') #(2386,928)
        self.labels = np.squeeze(data['Y']).astype('int').reshape(((len(data['Y'])), -1))  #(2386,)

    def __len__(self):
        return 2386

    def __getitem__(self, idx):
        # if self.view == 2:
        #     return [torch.from_numpy(
        #         self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

        return [torch.Tensor(self.view1[idx]), torch.Tensor(
            self.view2[idx]), torch.Tensor(self.view3[idx]), torch.Tensor(
            self.view4[idx]), torch.Tensor(self.view5[idx]), torch.Tensor(self.view6[idx])], torch.Tensor(self.labels[idx]), torch.Tensor(np.array(idx)).long()
