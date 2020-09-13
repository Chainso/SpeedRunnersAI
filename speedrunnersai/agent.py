from hlrl.torch.agents import OffPolicyAgent
from collections import OrderedDict

class Agent(OffPolicyAgent):
    """
    Creates the SpeedRunners agent, a modified OffPolicyAgent since the
    environment already returns torch tensors.
    """
    def transform_state(self, state):
        """
        Creates the dictionary of algorithm inputs from the env state
        """
        return OrderedDict({
            "state": state
        })
    
    def transform_reward(self, reward):
        """
        Transforms the reward of an environment step.
        """
        return reward

    def transform_terminal(self, terminal):
        """
        Transforms the terminal of an environment step.
        """
        return terminal

    def transform_action(self, action):
        """
        Transforms the action of the algorithm output to be usable with the
        environment.
        """
        return action