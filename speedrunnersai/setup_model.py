
import torch

from torch import nn
from hlrl.core.logger import TensorboardLogger
from hlrl.torch.agents import TorchRLAgent
from hlrl.torch.algos import TorchRLAlgo, RainbowIQN
from hlrl.torch.policies import LinearPolicy, Conv2dPolicy
from argparse import Namespace
from typing import Tuple

from speedrunnersai.speedrunners import SpeedRunnersEnv
from speedrunnersai.agent import Agent

class Autoencoder(Conv2dPolicy):
    """
    The convolution-based autoencoder for the network.
    """
    def forward(self, inp):
        """
        Returns the policy output for the input
        """
        out = super().forward(inp)
        return out.view(out.shape[0], -1)

def setup_model(args: Namespace) -> Tuple[
        SpeedRunnersEnv, TorchRLAlgo, TorchRLAgent
    ]:
    """
    Returns the setup environment, algorithm and agent using the arguments
    given.

    Args:
        args (Namespace)
    """
    # Environment
    env = SpeedRunnersEnv(
        args.episode_length, args.state_size, args.grayscale, args.stacked_frames,
        args.window_size, args.device, args.read_memory
    )

    # Algorithm
    logger = args.logs_path
    logger = None if logger is None else TensorboardLogger(logger)
    
    # Initialize SAC
    activation_fn = nn.ReLU
    optim = lambda params: torch.optim.Adam(params, lr=args.lr)

    # Setup networks
    qfunc = LinearPolicy(
        1024, env.action_space[0], args.hidden_size, 1,
        activation_fn
    )

    autoencoder = Autoencoder(
        (env.state_space[-1], 32, 64, 128, 128, 64),
        (5, 5, 3, 3, 3),
        (2, 2, 2, 2, 1),
        (0, 0, 0, 0, 0),
        activation_fn
    )

    algo = RainbowIQN(
        args.hidden_size, autoencoder, qfunc, args.discount, args.polyak,
        args.n_quantiles, args.embedding_dim, args.huber_threshold,
        args.target_update_interval, optim, optim, logger
    )
    algo = algo.to(args.device)

    if args.load_path is not None:
        algo.load(args.load_path)

    agent = Agent(
        env, algo, logger=logger, device=args.device
    )

    return env, algo, agent