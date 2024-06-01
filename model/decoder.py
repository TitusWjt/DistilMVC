import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self,
                 decoder_dim,
                 activation='relu',
                 batchnorm=True):
        super(Decoder, self).__init__()

        self._dim = len(decoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        #decoder_dim = [i for i in reversed(decoder_dim)]
        decoder_layers = []
        for i in range(self._dim-1):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(
            nn.Linear(decoder_dim[-2], decoder_dim[-1]))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        return self._decoder(z)

