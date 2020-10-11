import torch

from torch.cuda import is_available as cuda_available
from argparse import ArgumentParser, Namespace

def get_args() -> Namespace:
    """
    Setups up all the relavant command line arguments, and returns the parsed
    arguments.
    """
    parser = ArgumentParser("SpeedRunnersAI running configuration.")

    # File args
    parser.add_argument(
        "-x", "--experiment_path", type=str,
        help="the path to save experiment results and models"
    )
    parser.add_argument(
        "--load_path", type=str,
        help="the path of the saved model to load"
    )
    
    # Environment Arguments
    parser.add_argument(
        "--episode_length", type=int, default=120,
        metavar="NUM_SECONDS", help="the duration of each episode in seconds"
    )
    parser.add_argument(
        "--action_delay", type=float, default=0.2,
        help="the time in seconds between each agent action"
    )
    parser.add_argument(
        "--state_size", type=int, nargs=2, default=(64, 64),
        metavar=("HEIGHT", "WIDTH"),
        help="the size of each state in (height width)"
    )
    parser.add_argument(
        "--rgb", action="store_true",
        help="if the images should be rgb instead of grayscale, default false"
    )
    parser.add_argument(
        "--stacked_frames", type=int, default=4, metavar="NUM_FRAMES",
        help="the number of frames in a row for each model input"
    )
    parser.add_argument(
        "--read_memory", action="store_true",
        help="if this process should attach to the game and read memory from "
            + "it, default false"
    )
    parser.add_argument(
        "--window_size", type=int, nargs=4, default=None,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="the region of the screen the game is in, in "
            + "(left top right bottom), by default your entire screen"
    )

    # Model parameter arguments
    parser.add_argument(
		"--device", type=torch.device,
        default="cuda" if cuda_available() else "cpu",
		help="the device tensors should be stored on, if not given, will use "
            + "cuda if available"
	)
    parser.add_argument(
        "--hidden_size", type=int, default=512,
        help="the size of each hidden layer"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="the number of layers before the output layers"
    )

    # Algorithm arguments
    parser.add_argument(
        "--exploration", choices=["rnd"],
        help="The type of exploration to use [rnd]"
    )
    parser.add_argument(
		"--discount", type=float, default=0.99,
		help="the next state reward discount factor"
	)
    parser.add_argument(
        "--polyak", type=float, default=5e-3,
        help="the polyak constant for the target network updates"
    )
    parser.add_argument(
		"--n_quantiles", type=float, default=64,
		help="the number of quantile samples for IQN"
	)
    parser.add_argument(
		"--embedding_dim", type=float, default=64,
		help="the dimension of the quantile distribution for IQN"
	)
    parser.add_argument(
		"--huber_threshold", type=float, default=1,
		help="the threshhold of the huber loss (kappa) for IQN"
	)
    parser.add_argument(
        "--target_update_interval", type=float, default=1,
        help="the number of training steps in-between target network updates"
    )
    parser.add_argument(
		"--lr", type=float, default=3e-4, help="the learning rate"
	)

    # Training/Playing arguments
    parser.add_argument(
		"--episodes", type=int, default=10000,
		help="the number of episodes to play for if playing"
	)
    parser.add_argument(
		"--batch_size", type=int, default=256,
		help="the batch size of the training set"
	)
    parser.add_argument(
		"--start_size", type=int, default=1024,
		help="the size of the replay buffer before training"
	)
    parser.add_argument(
        "--training_steps", type=int, default=100000000,
        help="the number of training steps to train for"
    )
    parser.add_argument(
		"--save_interval", type=int, default=5000,
		help="the number of batches in between saves"
	)

    # Agent arguments
    parser.add_argument(
		"--decay", type=float, default=0.99,
		help="the gamma decay for the target Q-values"
	)
    parser.add_argument(
		"--n_steps", type=int, default=20,
		help="the number of decay steps"
	)
    parser.add_argument(
        "--silent", action="store_true",
        help="will run without standard output from agents"
    )

    # Experience Replay args
    parser.add_argument(
		"--er_capacity", type=float, default=7000,
		help="the maximum amount of episodes in the replay buffer"
	)
    parser.add_argument(
		"--er_alpha", type=float, default=0.6,
		help="the alpha value for PER"
	)
    parser.add_argument(
		"--er_beta", type=float, default=0.4,
		help="the beta value for PER"
	)
    parser.add_argument(
		"--er_beta_increment", type=float, default=1e-6,
		help="the increment of the beta value on each sample for PER"
	)
    parser.add_argument(
		"--er_epsilon", type=float, default=1e-3,
		help="the epsilon value for PER"
	)

    args = parser.parse_args()

    return args