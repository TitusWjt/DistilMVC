from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

from data.dataset.BDGP import BDGP
from data.dataset.CCV import CCV
from data.dataset.Fashion import Fashion
from data.dataset.Scene import Scene
from data.dataset.LandUse import LandUse
from data.dataset.MNIST_USPS import MNIST_USPS
from data.dataset.Caltech import Caltech


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech101-20":
        dataset = Caltech('./data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 2386
        class_num = 20
    elif dataset == "LandUse-21":
        dataset = LandUse('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 2100
        class_num = 21
    elif dataset == "Scene-15":
        dataset = Scene('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
