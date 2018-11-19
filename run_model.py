import torch

from configparser import ConfigParser

from model import Model
from screen_viewer import ScreenViewer
from actor import Actor
from time import time
def run_model(model, screen_viewer, actor, load_path, cuda):
    if(cuda):
        model = model.cuda()

    model.load(load_path)

    screen_viewer.Start()

    last = None

    while(True):
        a = time()
        screen = screen_viewer.GetNewScreen(last)
        print("Time 1:", time() - a)
        if(cuda):
            screen = torch.cuda.FloatTensor([screen]).permute(0, 3, 1, 2)
        else:
            screen = torch.FloatTensor([screen]).permute(0, 3, 1, 2)

        action = model(screen)[0]

        #actor.act(action)
        last = screen
        print("Time 2:", time() - a)
    screen_viewer.Stop()

def read_config():
    """
    Reads in the window sizes for the screen viewer from the config file
    """
    config = ConfigParser()

    # Read the config file, make sure not to re-name
    config.read("config.ini")

    return (config["Window Size"], config["SpeedRunners Config"])

if(__name__ == "__main__"):
    model = Model()

    window_size, speedrunners_config = read_config()
    window_size = (int(window_size["WIDTH"]), int(window_size["HEIGHT"]),
                   int(window_size["DEPTH"]))
    speedrunners_size = (int(speedrunners_config["WIDTH"]),
                         int(speedrunners_config["HEIGHT"]))
    screen_viewer = ScreenViewer(speedrunners_size, window_size)

    actor = Actor()
    load_path = "./Trained Models/model-15.torch"
    cuda = False

    run_model(model, screen_viewer, actor, load_path, cuda)
