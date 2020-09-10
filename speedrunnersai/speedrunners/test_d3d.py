if __name__ == "__main__":
    import d3dshot
    import torch
    import numpy as np
    from time import time, sleep
    from torch.nn.functional import interpolate
    from PIL import Image

    d3d = d3dshot.create(capture_output="pytorch_float_gpu")

    d3d.capture(target_fps=60)
    sleep(1)
    d3d.stop()
    """
    capture_time = 1

    d3d.capture(target_fps=60)

    last_frame = None
    frame = None
    frames = 0
    count = 0

    while frame is None:
        frame = d3d.get_latest_frame()

    frame = d3d.get_frame_stack([0], stack_dimension="first")

    last_frame = frame

    a = time()

    while time() - a < capture_time:
        while last_frame.equal(frame):
            count += 1
            print(count)
            frame = d3d.get_frame_stack([0], stack_dimension="first")

        last_frame = frame

        b = frame.permute(0, 3, 1, 2)
        b = interpolate(frame, size=(128, 128), mode="bilinear")
        print(frame.shape, b.shape)
        frames += 1


    d3d.stop()

    print("Captured {} frames, {} FPS".format(frames, frames/capture_time))
    """

    formula = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    formula_stack = torch.from_numpy(formula)

    frame_stack = d3d.get_frame_stack([0], stack_dimension="first")

    formula_stack = formula_stack.to(frame_stack.device)
    #formula_stack = formula_stack.repeat(frame_stack.shape[0], 1)

    print(formula_stack.shape)
    print(frame_stack.shape)

    frame_stack = torch.einsum('nhwc,c->nhw', frame_stack, formula_stack)
    frame_stack = frame_stack.unsqueeze(1)


    frame_stack = interpolate(frame_stack, size=(128, 128), mode="bilinear")

    frame = frame_stack.squeeze(1)[0] * 255
    frame = frame.cpu().byte().numpy()

    print(frame_stack.shape)
    img = Image.fromarray(frame, 'L')
    img.show()
    #print(frame)
    