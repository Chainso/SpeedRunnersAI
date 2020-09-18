def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

if __name__ == "__main__":
    import torch
    import gc

    from time import time

    from speedrunnersai.speedrunners import SpeedRunnersEnv
    from speedrunnersai.config.argument_parser import get_args

    args = get_args()

    env = SpeedRunnersEnv(
        args.episode_length, args.state_size, not args.rgb,
        args.stacked_frames, args.window_size, args.device, args.read_memory
    )

    env.start()

    start = time()
    frames = []
    while(len(frames) < 100):
        env.reset()
        frame = env.frame_handler.get_frame_stack(range(args.stacked_frames), "first")

        frames.append(frame)
        #if (len(frames) > 1):
            #print(frames[len(frames) - 2][0, 0].shape)
            #print(frames[len(frames) - 2][0, 1].shape)
            #print(frames[len(frames) - 2][0, 0].equal(frames[len(frames) - 1][0, 1]))
        print("Frame {}: State size {}, Current memory allocated {}".format(
            len(frames),
            sizeof_fmt(frame.element_size() * frame.nelement()),
            sizeof_fmt(torch.cuda.memory_stats(frame.device)["allocated_bytes.all.current"])
        ))

    env.stop()

    total_time = time() - start

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                total_size += obj.element_size() * obj.nelement()
        except:
            pass


    print("Memory of allocated tensors: {}", sizeof_fmt(total_size))
    print("Grabbed {} frames in {} time, {} FPS".format(
        len(frames), total_time, len(frames) / total_time
    ))
