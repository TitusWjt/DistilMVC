from model.encoder import Encoder
from model.decoder import Decoder
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self,
                 config,
                 view, dims, class_num):
        super(Autoencoder, self).__init__()
        self._config = config
        self.view = view
        self.dims = dims
        self.class_num = class_num


        if self._config['Encoder']['arch'][-1] != self._config['Decoder']['arch'][0]:
            raise ValueError('Inconsistent latent dim!')

        self.encoders_list = []
        self.decoders_list = []
        for v in range(self.view):
            self.encoders_list.append(Encoder([self.dims[v]] + self._config['Encoder']['arch'], self._config['Encoder']['function'], self._config['Encoder']['batchnorm']))
            self.decoders_list.append(Decoder(self._config['Decoder']['arch'] + [self.dims[v]],  self._config['Decoder']['function'], self._config['Decoder']['batchnorm']))
        self.encoders = nn.ModuleList(self.encoders_list)
        self.decoders = nn.ModuleList(self.decoders_list)

    def forward(self, xs):
        zs = []
        rs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            r = self.decoders[v](z)
            zs.append(z)
            rs.append(r)
        return  zs, rs

    def to_device(self, device):
        """ to cuda if gpu is used """
        for v in range(self.view):
            self.encoders[v].to(device)
            self.decoders[v].to(device)


