import torch
import torch.nn as nn
import numpy as np

def discount(rewards, decay):
    """
    Returns the discounted returns of the reward vectors.
    """
    ret = 0
    returns = []

    for reward in rewards[::-1]:
        ret = reward + decay * ret
        returns.append(ret)

    return np.array(returns[::-1])

def normalize(vector, epsilon):
    """
    Normalizes the given vector.
    """
    return (vector - vector.mean()) / (vector.std() + epsilon)

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.05):
        """
        Creates a module that adds gaussian noise to its inputs.
        """
        nn.Module.__init__(self)

        self.mean = mean
        self.std = std

    def forward(self, inp):
        noise = self.mean + self.std * torch.randn_like(inp)
        noise = noise.detach()

        return inp + noise

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, mean=0, std=0.05, bias=True):
        """
        Creates a module that adds gaussian noise to its inputs.
        """
        nn.Module.__init__(self)

        self.linear = nn.Linear(in_features, out_features, bias)
        self.noise = GaussianNoise(mean, std)

    def forward(self, inp):
        lin = self.linear(inp)
        noise = self.noise(lin)

        out = lin + noise

        return inp + noise

class LSTM(nn.LSTM):
    def __init__(self, minibatch_size, *args, **kwargs):
        """
        Creates an LSTM with the given arguments. The arguments and dictionary
        of arguments are the same as the ones in PyTorch's nn.LSTM module

        minibatch_size : The minibatch size of the LSTM
        *args : The arguments for the LSTM
        **kwargs : A dictionary of named arguments for the LSTM
        """
        # Call nn.LSTM's init with the given arguments
        nn.LSTM.__init__(self, *args, **kwargs)

        # Get the minibatch size
        self.minibatch_size = minibatch_size

        # Get the device of the hidden layers
        self.device = device

        # Create the hidden layer with 0 initialization
        self.hidden_layers = [torch.zeros(self.num_layers,
                                          self.minibatch_size,
                                          self.hidden_size),
                              torch.zeros(self.num_layers,
                                          self.minibatch_size,
                                          self.hidden_size)]

    def forward(self, inp):
        """
        Runs the input through the LSTM and returns the output

        inp : The input to the LSTM module
        """
        # Get the last n hidden just in case
        hidden = [self.hidden_layers[i][:, -inp.size()[1]:, :].to(inp.device)
                  for i in range(len(self.hidden_layers))]

        # Get the output and new hiddens state
        out, hidden = nn.LSTM.forward(self, inp, hidden)

        # Set the new hidden state
        self.hidden_layers = [h.detach().to(inp.device)
                                for h in hidden]

        return out

    def reset_hidden(self):
        """
        Resets the hidden state of the network to 0
        """
        self.hidden_layers = [torch.zeros(self.num_layers,
                                          self.minibatch_size,
                                          self.hidden_size),
                              torch.zeros(self.num_layers,
                                          self.minibatch_size,
                                          self.hidden_size)]

        self.hidden_layers = [h.to(torch.device(self.device))
                                for h in self.hidden_layers]

class QuantileLayer(nn.Module):
    def __init__(self, quantile_dim, output_dim):
        nn.Module.__init__(self)

        self.quantile_fc = nn.Linear(quantile_dim, output_dim)
        self.relu = nn.ReLU(-1)

    def forward(self, inp, num_quantiles=32):
        quantiles = torch.rand(self.num_quantiles * len(inp), 1,
                               device = inp.device)
        quantile_dist = quantiles.repeat(1, self.quantile_dim)

        qrange = torch.range(1, self.quantile_dim + 1, device = inp.device)

        quantile_dist = qrange * np.pi * quantile_dist
        quantile_dist = torch.cos(quantile_dist)
        quantile_dist = self.quantile_fc(quantile_dist)
        quantile_dist = self.relu(quantile_dist)

        rep_inp = inp.repeat(self.num_quantiles, 1)
        quantile_dist = rep_inp * quantile_dist

        quantile_values = self.fc(quantile_dist)
        quantile_values = quantile_values.view(self.num_quantiles,
                                               len(inp), self.act_n)

        return quantile_values, quantiles