import torch

def do_mul(t1, t2):
    fin = torch.matmul(t1, t2)
    return fin

def do_einsum(tensor1, tensor2):
    return torch.einsum('nhwc,c->nhw', tensor1, tensor2)

def do_compute(tensor1, tensor2):
    fin1 = do_mul(tensor1, tensor2)
    fin2 = do_einsum(tensor1, tensor2)
    assert fin1.equal(fin2)
    return fin1

if __name__ == "__main__":
    import d3dshot
    from time import time, sleep
    from torch.nn.functional import interpolate
    from PIL import Image

    from speedrunnersai.speedrunners.vision import FrameHandler

    d3d = d3dshot.create(capture_output="pytorch_float_gpu")

    formula = torch.FloatTensor([0.2125, 0.7154, 0.0721]).to("cuda")
    grayscale_transform = lambda frame: do_mul(frame, formula).unsqueeze(1)
    interpolate_transform = lambda frame: interpolate(
        frame, size=(128, 128), mode="bilinear"
    )
    
    transforms = [grayscale_transform]#, interpolate_transform]
    capture_time = 10
    frame_handler = FrameHandler(d3d, transforms=transforms)



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
