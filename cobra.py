import torch
import torch.nn as nn

class COBRA(nn.Module):
    def __init__(self, state_space, act_n, z_dim, batch_size, optim_params,
                 device = "cpu"):
        nn.Module.__init__(self)

        self.encoder = Encoder(state_space, z_dim, batch_size, device)
        self.decoder = Decoder(state_space, z_dim, device)

        self.enc_optim = torch.optim.Adam(self.encoder.parameters(), *optim_params)
        self.dec_optim = torch.optim.Adam(self.decoder.parameters(), *optim_params)

    def forward(self, inp):
        pass

    def step(self, inp, stochastic=True):
        pass

class Encoder(nn.Module):
    def __init__(self, state_space, z_dim, batch_size, device = "cpu"):
        nn.Module.__init__(self)

    def forward(self, inp):
        pass

    def step(self, inp, stochastic=True):
        pass

class Decoder(nn.Module):
    def __init__(self, state_space, z_dim, device = "cpu"):
        nn.Module.__init__(self)

    def forward(self, inp):
        pass

    def step(self, inp, stochastic=True):
        pass