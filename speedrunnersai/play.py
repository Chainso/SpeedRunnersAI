if __name__ == "__main__":
    import torch.multiprocessing as mp


    from speedrunnersai.setup_model import setup_model
    from speedrunnersai.config.argument_parser import get_args

    args = get_args()

    env, algo, agent = setup_model(args)

    algo.eval()

    env.start()
    agent.play(args.episodes)
    env.stop()