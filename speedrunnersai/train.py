if __name__ == "__main__":
    import torch.multiprocessing as mp

    from hlrl.core.agents import IntrinsicRewardAgent
    from hlrl.core.distributed import ApexRunner
    from hlrl.core.logger import TensorboardLogger
    from hlrl.core.common.functional import compose
    from hlrl.torch.experience_replay import TorchPER
    from pathlib import Path

    from speedrunnersai.setup_model import setup_model
    from speedrunnersai.config.argument_parser import get_args

    args = get_args()

    logs_path = None
    save_path = None

    if args.experiment_path is not None:
        logs_path = Path(args.experiment_path, "logs")
        logs_path.mkdir(parents=True, exist_ok=True)
        logs_path = str(logs_path)

        save_path = Path(args.experiment_path, "models")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = str(save_path)

    env, algo, agent_builder = setup_model(args)

    algo.create_optimizers()

    algo.train()
    algo.share_memory()

    # Finish setting up agent
    if args.exploration == "rnd":
        agent_builder = compose([agent_builder, IntrinsicRewardAgent])
    
    experience_replay = TorchPER(
        args.er_capacity, args.er_alpha, args.er_beta,
        args.er_beta_increment, args.er_epsilon
    )

    base_agents_logs_path = None
    if logs_path is not None:
        base_agents_logs_path = logs_path + "/train-agent-"

    # Can't reinitialize CUDA on windows, so no parallelization in this format
    # Also some pickle problems on windows, need to investigate
    """
    if str(args.device) == "cpu":
        done_event = mp.Event()

        agent_queue = mp.Queue()
        sample_queue = mp.Queue()
        priority_queue = mp.Queue()

        learner_args = (
                algo, done_event, args.training_steps, sample_queue,
                priority_queue, save_path, args.save_interval
        )

        worker_args = (
                experience_replay, done_event, agent_queue, sample_queue,
                priority_queue, args.batch_size, args.start_size,
        )

        agents = []
        agent_train_args = []

        # May be able to use virtual displays to parallelize in the future
        for i in range(1):
            agent_logger = None
            if base_agents_logs_path is not None:
                agent_logs_path = base_agents_logs_path + str(i + 1)
                agent_logger = TensorboardLogger(agent_logs_path)

            agents.append(agent_builder(logger=agent_logger))
            agent_train_args.append((
                done_event, args.decay, args.n_steps, agent_queue
            ))

        runner = ApexRunner(done_event)

        env.start()
        runner.start(learner_args, worker_args, agents, agent_train_args)
        env.stop()
    else:
    """
    agent_logger = None
    if base_agents_logs_path is not None:
        agent_logs_path = base_agents_logs_path + "0"
        agent_logger = TensorboardLogger(agent_logs_path)

    agent = agent_builder(logger=agent_logger)

    env.start()

    agent.train(
        args.training_steps + args.start_size, args.decay, args.n_steps,
        experience_replay, algo, args.batch_size, args.start_size, save_path,
        args.save_interval
    )

    env.stop()
