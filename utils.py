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
