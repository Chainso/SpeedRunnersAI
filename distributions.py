import torch.nn as nn

class GaussianMixtureModel(nn.Module):
    def __init__(self, num_mixture):
        """
        A gaussian mixture model (computes the means and log standard
        deviations)
        """
        self.num_mixture = num_mixture

class MultiCategoricalDistribution(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        """
        Creates a multi-categorical distribution with the given number of
        inputs and outputs

        num_inputs : The number of inputs of the distribution
        num_outputs : The number of outputs of the distribution
        """
        nn.Module.__init__(self)

        # The layer for the distribution
        self.output = nn.Linear(num_inputs, num_outputs)

        # The loss function
        #self.loss_func = nn.BCEWithLogitsLoss()

        # Temporary for tanh
        self.loss_func = nn.MSELoss()

    def forward(self, inp):
        """
        Returns the raw distribution output for the given inputs
        """
        return self.output(inp)

    def distribution(self, inp):
        """
        Returns the probability distribution for the given inputs, which is
        the raw distribution with an additional sigmoid layer
        """
        # Temporary for tanh
        return nn.Tanh()(self(inp))
        #return nn.Sigmoid()(self(inp))

    def calculate_loss(self, inp, out):
        """
        Calculates the losses for the given input and target output

        inp : The input
        out : The target output
        """
        # Get the predicted output
        #pred_out = self(inp)

        # Temporary for tanh
        pred_out = self.distribution(inp)

        # Calculate the loss
        return self.loss_func(pred_out, out)
