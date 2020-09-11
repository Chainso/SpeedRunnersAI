if __name__ == "__main__":
    import torch

    from time import time
    from speedrunnersai.speedrunners import SpeedRunnersEnv

    max_time = 120
    res_shape = (128, 128)
    grayscale = True
    stacked_frames = 1
    window_size = None
    device = "cuda"
    read_mem = False

    env = SpeedRunnersEnv(
        max_time, res_shape, grayscale, stacked_frames, window_size, device,
        read_mem
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

    
    print("Total time: {}, {} FPS".format(total_time, frames/total_time))