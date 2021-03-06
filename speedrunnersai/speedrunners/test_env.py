if __name__ == "__main__":
    import torch

    from time import time

    from speedrunnersai.speedrunners import SpeedRunnersEnv
    from speedrunnersai.config.argument_parser import get_args

    args = get_args()
    print(args)

    env = SpeedRunnersEnv(
        args.episode_length, args.state_size, not args.rgb, args.stacked_frames,
        args.window_size, args.device, args.read_memory
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
