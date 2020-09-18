if __name__ == "__main__":
    import torch.multiprocessing as mp

    from hlrl.core.agents import AgentPool
    from hlrl.core.trainers import Worker
    from hlrl.torch.experience_replay import TorchPER

    from speedrunnersai.setup_model import setup_model
    from speedrunnersai.config.argument_parser import get_args

    args = get_args()

    env, algo, agent = setup_model(args)

    algo.create_optimizers()

    algo.train()
    algo.share_memory()

    experience_queue = mp.Queue()

    experience_replay = TorchPER(
        args.er_capacity, args.er_alpha, args.er_beta, args.er_beta_increment,
        args.er_epsilon
    )

    env.start()

    agent.train(
        args.episodes, args.decay, args.n_steps, experience_replay, algo,
        args.batch_size, args.start_size, args.save_dir, args.save_interval
    )

    #agents = [agent]

    #agent_pool = AgentPool(agents)
    #agent_procs = agent_pool.train_process(
        #args.episodes, args.decay, args.n_steps, experience_queue
    #)

    # Start the worker for the model
    #worker = Worker(algo, experience_replay, experience_queue)
    #worker.train(
        #agent_procs, args.batch_size, args.start_size, args.save_dir,
        #args.save_interval
    #)
    
    env.stop()