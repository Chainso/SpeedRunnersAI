
import torch

from torch import nn
from hlrl.core.logger import Logger, TensorboardLogger
from hlrl.core.agents import OffPolicyAgent
from hlrl.core.common.functional import compose
from hlrl.torch.agents import TorchRLAgent
from hlrl.torch.algos import TorchRLAlgo, RainbowIQN, RND
from hlrl.torch.policies import LinearPolicy, Conv2dPolicy
from argparse import Namespace
from typing import Tuple, Callable
from functools import partial
from pathlib import Path
from copy import deepcopy

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
        SpeedRunnersEnv, Callable[[Logger], TorchRLAlgo], TorchRLAgent
    ]:
    """
    Returns the setup environment, algorithm and a builder for an agent using
    the arguments given.

    Args:
        args (Namespace): The input arguments to setup the model, environment
            and agent.
    """
    logs_path = None

    if args.experiment_path is not None:
        logs_path = Path(args.experiment_path, "logs")
        logs_path.mkdir(parents=True, exist_ok=True)
        logs_path = str(logs_path)

    # Environment
    env = SpeedRunnersEnv(
        args.episode_length, args.state_size, args.action_delay, not args.rgb,
        args.stacked_frames, args.window_size, args.device, args.read_memory
    )

    # The algorithm logger
    algo_logger = (
        None if logs_path is None else TensorboardLogger(logs_path + "/algo")
    )
    
    # Initialize SAC
    activation_fn = nn.ELU
    optim = partial(torch.optim.Adam, lr=args.lr)

    # Setup networks
    qfunc = LinearPolicy(
        1024, env.action_space[0], args.hidden_size, args.num_layers,
        activation_fn
    )

    autoencoder = Autoencoder(
        (env.state_space[-1], 32, 64, 64),
        (8, 4, 3),
        (4, 2, 1),
        (0, 0, 0),
        activation_fn
    )

    algo = RainbowIQN(
        args.hidden_size, autoencoder, qfunc, args.discount, args.polyak,
        args.n_quantiles, args.embedding_dim, args.huber_threshold,
        args.target_update_interval, optim, optim, device=args.device,
        logger=algo_logger
    )

    if args.exploration == "rnd":
        rnd_network_autoencoder = deepcopy(autoencoder)
        rnd_network_linear = LinearPolicy(
            env.state_space[0], args.hidden_size, args.hidden_size,
            args.num_layers + 2, activation_fn
        )
        rnd_network = nn.Sequential(rnd_network_autoencoder, rnd_network_linear)
        
        rnd_target_autoencoder = deepcopy(autoencoder)
        rnd_target_linear = LinearPolicy(
            env.state_space[0], args.hidden_size, args.hidden_size,
            args.num_layers, activation_fn
        )
        rnd_target = nn.Sequential(rnd_target_autoencoder, rnd_target_linear)

        algo = RND(algo, rnd_network, rnd_target, optim)

    algo = algo.to(args.device)

    if args.load_path is not None:
        algo.load(args.load_path)

    agent_builder = partial(OffPolicyAgent, env, algo, silent=args.silent)
    agent_builder = compose(agent_builder, Agent)

    return env, algo, agent_builder