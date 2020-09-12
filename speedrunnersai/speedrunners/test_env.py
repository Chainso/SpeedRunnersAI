from configparser import ConfigParser

def read_config():
    """
    Reads the config file to obtain the settings for the recorder, the
    window size for the training data and the game bindings
    """
    config = ConfigParser()

    # Read the config file, make sure not to re-name
    config.read("../config/config.ini")

    return config["Environment"], config["AI"]                                          

if __name__ == "__main__":
    import torch

    from time import time
    from speedrunnersai.speedrunners import SpeedRunnersEnv
    from PIL import Image

    env_config, ai_config = read_config()
    print(bool(env_config["READ_MEM"]))
    """
    env = SpeedRunnersEnv(
        int(env_config["MAX_TIME"]),
        (int(env_config["HEIGHT"]), int(env_config["WIDTH"])),
        bool(env_config["GRAYSCALE"]), int(env_config["STACKED_FRAMES"]),
        (int(env_config["LEFT"]), int(env_config["TOP"]),
            int(env_config["RIGHT"]), int(env_config["BOTTOM"])),
        ai_config["DEVICE"], bool(env_config["READ_MEM"])
    )
    
    frames = 300

    env.start()
    cur_time = time()
    for i in range(frames):
        env.reset()
        action = env.sample_action()
        env.step(action)

    total_time = time() - cur_time
    env.stop()

    print(env.state.shape)
    print(env.match.players[0])
    print(env.match.players[0].speed())
    print("Total time: {}, {} FPS".format(total_time, frames/total_time))

    frame = env.state.squeeze(1)[0] * 255
    frame = frame.byte().cpu().numpy()
    img = Image.fromarray(frame, "L")
    img.show()
    """