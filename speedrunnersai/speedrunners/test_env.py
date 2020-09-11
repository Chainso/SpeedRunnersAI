if __name__ == "__main__":
    import torch

    from time import time, sleep
    from PIL import Image

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

    total_time = time() - cur_time
    env.stop()

    
    ###
    last_second = env.frame_handler.get_frame_stack(range(60), "first").flip(0)
    last_second = last_second.squeeze(1) * 255
    last_second = last_second.byte()

    uniques = torch.unique_consecutive(last_second, dim=0)

    print("From: {} to {}".format(last_second.shape, uniques.shape))
    print("Total time: {}, {} FPS".format(total_time, frames/total_time))
    
    last_second = last_second.cpu().numpy()
    uniques = uniques.cpu().numpy()
    fp_out = "./speedrunners.gif"

    img, *imgs = [Image.fromarray(arr) for arr in uniques]

    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=10, loop=0)
