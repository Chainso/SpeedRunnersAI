import torch
import torch.nn as nn
import numpy as np

from utils import LSTM, NoisyLinear, QuantileLayer

RANDOM_SEED = 999
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Model(nn.Module):
    """
    Recurrent IQN with R2D3 features

    IQN: https://arxiv.org/pdf/1806.06923.pdf
    R2D2: https://openreview.net/pdf?id=r1lyTjAqYX
    R2D3: https://arxiv.org/abs/1909.01387
    """
    def __init__(self, state_space, act_n, quantile_dim, hidden_dim,
                 num_hidden):
        nn.Module.__init__(self)

        self.state_space = state_space
        self.act_n = act_n
        self.quantile_dim = quantile_dim

        self.conv = ConvNet(state_space[-1], 32, lambda x: 2 * x)

        self.lstm = nn.LSTM(256, quantile_dim)
        # If input size is 256 after CNN
        self.quantile_layer = QuantileLayer(quantile_dim, hidden_dim)

        self.advantage = nn.Seqential(
            NoisyLinear(hidden_dim, hidden_dim, 0, 0.05),
            nn.ReLU(-1),
            NoisyLinear(hidden_dim, act_n, 0, 0.05)
        )

        self.advantage = nn.Seqential(
            NoisyLinear(hidden_dim, hidden_dim, 0, 0.05),
            nn.ReLU(-1),
            NoisyLinear(hidden_dim, 1, 0, 0.05)
        )

    def step(self, state, num_quantiles=32, greedy=False):
        q_vals, quantile_values, quantiles = self(state, num_quantiles)

        if(greedy):
            action = q_vals.max(1)
        else:
            action = nn.Categorical(logits = q_vals)
            action = action.sample()

        action = action.item()

        return action

    def forward(self, state, num_quantiles=32):
        """
        Returns the quantile values and quantiles for the given state
        """
        cnn = cnn.view(len(state), -1)

        quantile_values, quantiles = self.quantile_layer(cnn, num_quantiles)

        advantage = self.advantage(quantile_values)
        value = self.value(quantile_values)

        q_vals = value + advantage - advantage.mean(1, keepdim=True)

        return q_vals, quantile_values, quantiles

    def train_batch(rollouts):
        states, actions, rewards, next_states, terminals = rollouts
        rewards = rewards.repeat

    def save(self, save_path):
        """
        Saves the model at the given save path

        save_path : The path to save the model at
        """
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        """
        Loads the model at the given load path

        load_path : The of the model to load
        """
        self.load_state_dict(torch.load(load_path))

class ConvNet(nn.Module):
    def __init__(self, inp_channels, init_channels, growth_func):
        nn.Module.__init__(self)

        num_layers = 4
        hidden_channels = [init_channels]
        for i in range(num_layers - 1):
            hidden_channels.append(growth_func(hidden_channels[-1]))

        self.convs = nn.Sequential(
            self._conv_layer(inp_channels, hidden_channels[0], 5, 2, 0),
            self._conv_layer(inp_channels, hidden_channels[0], 3, 2, 0),
            self._conv_layer(inp_channels, hidden_channels[0], 3, 2, 0),
            self._conv_layer(inp_channels, hidden_channels[0], 3, 1, 0),
        )

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride,
                    padding):
        return nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(-1)
        )

    def forward(self, inp):
        return self.convs(inp)