from hlrl.torch.agents import TorchRLAgent
from collections import OrderedDict

class Agent(TorchRLAgent):
    """
    Wraps the SpeedRunners agent, a modified TorchRLAgent wrapper since the
    environment already returns torch tensors.
    """
    def transform_state(self, state):
        """
        Creates the dictionary of algorithm inputs from the env state
        """
        return OrderedDict({
            "state": state
        })