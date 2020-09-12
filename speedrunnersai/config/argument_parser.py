from torch.cuda import is_available
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser("SpeedRunnersAI running configuration.")

    # Environment Arguments
    parser.add_argument(
        "--max_time", type=int, default=120,
        metavar="NUM_SECONDS", help="the duration of each episode in seconds"
    )
    parser.add_argument(
        "--state_size", type=int, nargs=2, default=(128, 128),
        metavar=("HEIGHT", "WIDTH"),
        help="the size of each state in (height width)"
    )
    parser.add_argument(
        "--grayscale", action="store_true",
        help="if the images should be converted to grayscale, default true"
    )
    parser.add_argument(
        "--stacked_frames", type=int, default=1,
        metavar="NUM_FRAMES",
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

    # AI arguments
    parser.add_argument(
        "--device", type=str, default="cuda" if is_available() else "cpu",
        metavar="DEVICE_NAME",
        help="the device tensors should be stored on, if not given, will use "
            + "cuda if available"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models", metavar="SAVE_DIR",
        help="the directory to save models to"
    )
    parser.add_argument(
        "--save_name", type=str, default="model", metavar="BASE_NAME",
        help="the base name to append the step counter to when saving models"
    )
    parser.add_argument(
        "--load_path", type=str, default=None, metavar="LOAD_PATH",
        help="the path to load a previously saved model from, default none"
    )

    args = parser.parse_args()

    return args