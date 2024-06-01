import itertools
import os
import numpy as np
import torch
import random
import argparse
from torch.utils.data import Dataset
from data.dataloader.dataloader import load_data
from model.autoencoder import Autoencoder
from model.byol import BYOL
from model.distilmvc import DistilMVC
from utils import yaml_config_hook


def main():
    #Load hyperparameters
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./configs/config_bdgp.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    #Load dataset
    dataset, dims, view, data_size, class_num = load_data(args.dataset_name)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


    autoencoder = Autoencoder(args.model_kwargs, view, dims, class_num)
    byol = BYOL(args.model_kwargs['Encoder']['arch'][-1], args.model_kwargs['predictor_hidden_dim'], class_num)
    distilmvc = DistilMVC(autoencoder, byol, view)
    optimizer = torch.optim.Adam(
                                 itertools.chain(distilmvc.autoencoder.parameters(), distilmvc.byol.parameters(),),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    distilmvc.to_device(args.device)
    distilmvc.train(args, data_loader, optimizer, dataset, data_size, class_num)




if __name__ == '__main__':
    main()