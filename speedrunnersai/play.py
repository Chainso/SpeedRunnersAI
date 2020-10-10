if __name__ == "__main__":
    import torch.multiprocessing as mp

    from pathlib import Path

    from speedrunnersai.setup_model import setup_model
    from speedrunnersai.config.argument_parser import get_args
    from hlrl.core.logger import TensorboardLogger

    args = get_args()

    logs_path = None

    if args.experiment_path is not None:
        logs_path = Path(args.experiment_path, "logs")
        logs_path.mkdir(parents=True, exist_ok=True)
        logs_path = str(logs_path)

    env, algo, agent_builder = setup_model(args)

    algo.eval()

    agent_logger = (
        None if logs_path is None
        else TensorboardLogger(logs_path + "/play-agent")
    )

    agent = agent_builder(logger=agent_logger)

    env.start()
    agent.play(args.episodes)
    env.stop()