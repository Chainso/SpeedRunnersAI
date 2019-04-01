import torch
import torch.nn as nn

from utils import LSTM

class Model2(nn.Module):
    def __init__(self, state_space, act_n, il_weight, device = "cpu"):
        nn.Module.__init__(self)

        self.state_space = state_space
        self.il_weight = il_weight
        self.device = torch.device(device)

        self.conv = nn.Sequential(
                self._conv_block(state_space[-1], 32, 5, 2, 0),
                self._conv_block(32, 32, 3, 2, 0),
                self._conv_block(32, 64, 3, 2, 0),
                self._conv_block(64, 64, 3, 2, 0),
                self._conv_block(64, 64, 3, 1, 0)
                )

        self.linear = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p = 0.5)
                )

        self.lstm = LSTM(10, 256, 256, 1)
        self.lstm_dropout = nn.Dropout(p = 0.5)

        self.policy = nn.Sequential(
                nn.Linear(256, act_n),
                nn.Sigmoid()
                )

        self.value = nn.Linear(256, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)

    def _conv_block(self, filt_in, filt_out, kernel, stride, padding):
        return nn.Sequential(
                nn.Conv2d(filt_in, filt_out, kernel, stride, padding),
                nn.ReLU(),
                nn.Dropout(p = 0.6)
                )

    def _bernoulli_loss(self, policy, actions):
        return actions * torch.log(policy) + (1 - actions) * torch.log(1 - policy)

    def forward(self, inp):
        conv = self.conv(inp)
        conv = conv.view(-1, 1024)

        linear = self.linear(conv)
        linear = linear.view(15, -1, 256)

        lstm = self.lstm(linear)
        lstm = lstm.view(-1, 256)
        lstm = self.lstm_dropout(lstm)

        policy = self.policy(lstm)

        actions = torch.distributions.Bernoulli(policy)
        actions = actions.sample()

        value = self.value(lstm)

        return actions, policy, value

    def step(self, inp):
        conv = self.conv(inp)
        conv = conv.view(-1, 1024)

        linear = self.linear(conv)
        linear = linear.view(1, -1, 256)

        lstm = self.lstm(linear)
        lstm = lstm.view(-1, 256)
        lstm = self.lstm_dropout(lstm)

        policy = self.policy(lstm)

        actions = torch.distributions.Bernoulli(policy)
        actions = actions.sample()

        actions = actions[0].detach().cpu().numpy()
        policy = policy[0].detach().cpu().numpy()

        value = self.value(lstm)
        value = value.item()

        return actions, policy, value

    def train_supervised(self, states, actions):
        self.lstm.reset_hidden()

        new_acts, policy, value = self(states)

        policy_loss = -self.il_weight * self._bernoulli_loss(policy, actions)

        policy_loss = policy_loss.mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.detach().numpy()

    def train_reinforce(self, rollouts):
        self.lstm.reset_hidden()

        states, acts, rewards, advs = [torch.from_numpy(tensor).to(self.device)
                                       for tensor in rollouts]

        actions, policy, value = self(states)

        policy_loss = -advs * self._bernoulli_loss(policy, acts)
        policy_loss = policy_loss.mean()

        value_loss = nn.MSELoss()(value, rewards)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()

    def reset_hidden_state(self):
        """
        Resets the hidden state for the LSTM
        """
        self.lstm.reset_hidden()

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