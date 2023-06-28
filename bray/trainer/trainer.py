class RemoteTrainer:
    """
    这个类用于在多个节点上训练模型，它会在多个节点上创建 Trainer 的实例，
    然后调用 train 函数
    """

    def __init__(self, use_gpu: bool = None, num_workers: int = None):
        """
        Args:
            use_gpu: 是否使用GPU
            num_workers: 训练的节点数
        """
        from horovod.ray import RayExecutor
        import math
        import ray

        total_cpus = ray.available_resources()["CPU"]
        total_gpus = ray.available_resources().get("GPU", 0)

        use_gpu = use_gpu if use_gpu else total_gpus > 0

        if not use_gpu:
            trainer_cpus = max(1, total_cpus * 3 // 4)
            num_workers = num_workers if num_workers else int(math.sqrt(trainer_cpus))
            cpus_per_worker = trainer_cpus // num_workers
        else:
            cpus_per_worker = 2
            num_workers = num_workers if num_workers else total_gpus

        print(
            f"Trainer start with {num_workers} {'GPU' if use_gpu else 'CPU'} workers, "
            + f"{cpus_per_worker} cpus per worker"
        )

        settings = RayExecutor.create_settings()
        self.executor = RayExecutor(
            settings,
            num_workers=num_workers,
            cpus_per_worker=cpus_per_worker,
            use_gpu=use_gpu,
        )
        self.executor.start()

    def train(self, train: callable, *args, **kwargs) -> list[any]:
        """
        在多个节点上执行训练函数
        Args:
            train: 训练函数
            *args: 训练函数的位置参数
            **kwargs: 训练函数的关键字参数
        Returns:
            训练函数在每个节点上的返回值，可以通过 ray.get() 获取
        """
        return self.executor.run_remote(train, args, kwargs)
