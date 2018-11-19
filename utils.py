import torch
import torch.nn as nn


def forward(inp, layers, activations):
    """
    Computes the output of the given network and activations

    layers : A list of layers for the network
    activations : A list of activation functions
    """
    assert len(layers) == len(activations)

    output = inp

    for layer, activation in zip(layers, activations):
        output = layer(output)

        if(activation is not None):
            output = activation(output)

    return output

def linear_network(inp_units, units):
    """
    Creates a linear network with the given number of units per layer and
    activation per layer

    units : The number of output units in each layer
    
    """
    tot_units = [inp_units] + units
    layers = []

    for i in range(len(tot_units) - 1):
        layers.append(nn.Linear(tot_units[i], tot_units[i + 1]))

    return nn.ModuleList(layers)

def conv_network(input_channels, out_channels, kernel_sizes, strides,
                 paddings):
    """
    Creates a convolution network with the given parameters

    input_channels : The number of channels in the input image
    out_channels : The number of output channels in each convolution layer
    kernel_sizes : The sizes of the kernels in each convolution layer
    strides : The strides of the convolution in each layer
    paddings : The paddings of the convolution in layer
    """
    tot_channels = [input_channels] + out_channels
    layers = []

    for i in range(len(tot_channels) - 1):
        layers.append(nn.Conv2d(tot_channels[i], tot_channels[i + 1],
                                kernel_sizes[i], strides[i], paddings[i]))

    return nn.ModuleList(layers)

def inverse_conv_network(input_channels, out_channels, kernel_sizes, strides,
                         paddings):
    """
    Creates an inverse convolution network with the given parameters

    input_channels : The number of channels in the input image
    out_channels : The number of output channels in each convolution layer
    kernel_sizes : The sizes of the kernels in each convolution layer
    strides : The strides of the convolution in each layer
    paddings : The paddings of the convolution in layer
    """
    tot_channels = [input_channels] + out_channels
    layers = []

    for i in range(len(tot_channels) - 1):
        layers.append(nn.ConvTranspose2d(tot_channels[i], tot_channels[i + 1],
                                         kernel_sizes[i], strides[i],
                                         paddings[i]))

    return nn.ModuleList(layers)

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
        # Get the output and new hiddens state
        out, hidden = nn.LSTM.forward(self, inp, self.hidden_layers)

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
