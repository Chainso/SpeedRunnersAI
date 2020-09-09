import torch
import torch.nn as nn
import numpy as np

class GaussianMixtureModel(nn.Module):
    """
    A guassian mixture model that outputs means and log standard deviations
    """
    def __init__(self, num_inputs, num_mixture, reparameterize):
        """
        Creates the gaussian mixture model (computes the means and log standard
        deviations) of a gaussian mixture.

        Args:
            num_inputs (int): The number of input units.
            num_mixture (int): The number of output units in the layer.
            reparameterize (bool): Will use the reparameterization trick if
                                   true.
        """
        nn.Module.__init__(self)

        self.num_mixture = num_mixture
        self.reparameterize = reparameterize

        self.means = nn.Linear(num_inputs, num_mixture)
        self.log_stds = nn.Linear(num_inputs, num_mixture)

    @staticmethod
    def kl_divergence(means, log_stds):
        """
        Returns the mean KL divergence of the means and standard deviations
        given with the normal distribution
        """
        kl_divergence = -0.5 * (log_stds - log_stds.exp() - means.pow(2) + 1)
        kl_divergence = kl_divergence.mean()

        return kl_divergence

    def forward(self, inp):
        """
        Returns the means the standard deviation of the mixture model on the
        input.
        """
        return self.means(inp), self.log_stds(inp)

    def mixture(self, inp, ret_means_stds = False):
        """
        Returns the gaussian mixture for the input.

        Args:
            inp (Tensor): The input to compute the mixture for.
            ret_means_stds (bool): Will return the means and log stds if true.
        """
        means, log_stds = self(inp)

        if(self.reparameterize):
            gaussian = torch.random.normal(self.num_mixture,device = inp.device)
            gaussian = means + log_stds.exp() * gaussian
        else:
            gaussian = torch.normal(means, log_stds.exp())

        if(ret_means_stds):
            return gaussian, means, log_stds
        else:
            return gaussian

    def log_gaussian(self, inp):
        """
        Returns the log gaussian of the input.
        """
        means, log_stds = self(inp)
        
        normalized_inp = (inp - means) * torch.exp(-log_stds)
        squared_norm = -0.5 * normalized_inp.pow(2)

        sum_log_stds = log_stds.sum(-1).view(-1)
        sum_log_stds += 0.5 * means.sum(-1) * np.log(2 * np.pi)

        log_prob = squared_norm - sum_log_stds

        return log_prob

    def calculate_loss(self, inp, out):
        """
        Calculates the loss for the given input
        """
        kl_loss = self.kl_divergence(inp)

class TanhGaussianMixtureModel(GaussianMixtureModel):
    """
    A gaussian mixture model that gets squashed with the tanh function.
    """
    def __init__(self, num_inputs, num_mixture, reparameterize):
        """
        Creates the gaussian mixture model (computes the means and log standard
        deviations) of a gaussian mixture, that then gets squashed by a tanh.

        Args:
            num_inputs (int): The number of input units.
            num_mixture (int): The number of output units in the layer.
            reparameterize (bool): Will use the reparameterization trick if
                                   true.
        """
        GaussianMixtureModel.__init__(self, num_inputs, num_mixture,
                                      reparameterize)

        # Using MSE loss with the outputs
        self.loss_func = nn.MSELoss()

    def mixture(self, inp, ret_means_stds = False):
        """
        Returns the gaussian mixture for the input, squashed by tanh.

        Args:
            inp (Tensor): The input to compute the mixture for.
            ret_means_stds (bool): Will return the means and log stds if true.
        """
        return nn.Tanh()(GaussianMixtureModel.mixture(self, inp,
                                                      ret_means_stds))

    def calculate_loss(self, inp, out):
        """
        Calculates the loss of the gaussian mixture model with respect to the
        output given.
        """
        mixture, means, log_stds = self.mixture(inp, True)

        kl_loss = GaussianMixtureModel.kl_divergence(means, log_stds)
        out_loss = self.loss_func(mixture, out)

        loss = kl_loss + out_loss

        return loss

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
