import torch

if __name__ == "__main__":
    import d3dshot
    from time import time, sleep
    from PIL import Image

    from hlrl.core.vision import WindowsFrameHandler
    from hlrl.core.vision.transforms import Grayscale, Interpolate

    d3d = d3dshot.create(capture_output="pytorch_float_gpu")

    transforms = [
        Grayscale(),
        Interpolate(size=(128, 128), mode="bilinear")
    ]

    capture_time = 10
    frame_handler = WindowsFrameHandler(d3d, transforms=transforms)


    # A higher target fps since transforms will slow down the capturing
    frame_handler.capture(target_fps=240)

    frame = None
    frames = 0

    last_frame = frame

    print("Starting!")
    a = time()

    while time() - a < capture_time:
        frame_handler.get_new_frame()
        frame = frame_handler.get_frame_stack([0], "first")

        frames += 1


    frame_handler.stop()

    print("Captured {} frames, {} FPS".format(frames, frames/capture_time))
    print(frame.shape)
    frame = frame.squeeze(1)[0] * 255
    frame = frame.cpu().byte().numpy()

    img = Image.fromarray(frame, 'L')
    img.show()
