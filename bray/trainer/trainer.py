class RemoteTrainer:
    """
    这个类用于在多个节点上训练模型，它会在多个节点上创建 Trainer 的实例，
    然后调用 train 函数
    """

    def __init__(self, num_workers: int, use_gpu: bool):
        """
        Args:
            num_workers: 训练的节点数
            use_gpu: 是否使用GPU
        """
        from horovod.ray import RayExecutor

        settings = RayExecutor.create_settings()
        self.executor = RayExecutor(
            settings,
            num_workers=num_workers,
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
