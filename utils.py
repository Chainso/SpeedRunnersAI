import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.05):
        nn.Module.__init__(self)

        self.mean = mean
        self.std = std

    def forward(self, inp):
        noise = self.mean + self.std * torch.randn_like(inp)
        noise = noise.detach()

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
        hidden = [self.hidden_layers[i][:, -len(inp):, :]
                  for i in range(len(self.hidden_layers))]

        # Get the output and new hiddens state
        out, hidden = nn.LSTM.forward(self, inp, hidden)

        # Set the new hidden state
        self.hidden_layers = [h.detach() for h in hidden]

        return out

    def reset_hidden(self):
        """
        Resets the hidden state of the network to 0
        """
        self.hidden_layers = (torch.zeros(self.num_layers,
                                          self.minibatch_size,
                                          self.hidden_size),
                              torch.zeros(self.num_layers,
                                          self.minibatch_size,
                                          self.hidden_size))
