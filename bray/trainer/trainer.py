class RemoteTrainer:
    """
    这个类用于在多个节点上训练模型，它会在多个节点上创建 Trainer 的实例，
    然后调用 train 函数
    """

    def __init__(
        self,
        use_gpu: bool = None,
        num_workers: int = None,
        cpus_per_worker: int = None,
        total_cpus_ratio: float = 0.75,
    ):
        """
        Args:
            use_gpu: 是否使用GPU，如果不指定则会自动判断
            num_workers: 训练的 worker 数量，如果不指定则会自动计算
            cpus_per_worker: 每个节点的CPU核心数，仅当 use_gpu 为 False 时有效
            total_cpus_ratio: 训练节点的总CPU核心数占用比例，仅当 use_gpu 为 False 时有效
        """
        from horovod.ray import RayExecutor
        import math
        import ray

        total_cpus = ray.available_resources()["CPU"]
        total_gpus = ray.available_resources().get("GPU", 0)

        use_gpu = use_gpu if use_gpu else total_gpus > 0

        if not use_gpu:
            trainer_cpus = max(1, int(total_cpus * total_cpus_ratio))
            if not num_workers:
                num_workers = int(math.sqrt(trainer_cpus))
            if not cpus_per_worker:
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

        def init_torch():
            import torch

            torch.set_num_interop_threads(cpus_per_worker)
            torch.set_num_threads(cpus_per_worker)

        ray.get(self.executor.run_remote(init_torch))

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
