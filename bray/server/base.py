class Server:
    """
    Server是一个有状态服务接受来自Client的调用，调用的顺序是：
    start -> tick -> tick -> ... -> stop
    """

    def __init__(self, *args, **kwargs):
        """
        初始化一个新的Server，当一局新的游戏开始之前时，会调用这个方法
        Args:
            *args: 位置参数，由RemoteServer传入
            **kwargs: 关键字参数，由RemoteServer传入
        """

    async def start(self, session, data: bytes) -> bytes:
        """
        开始一局新的游戏，由Client调用，请在这里初始化游戏状态
        Args:
            session: 当前的游戏会话，由Client传入
            data: 一个任意的字节串，由Client传入，通常是游戏状态
        Returns:
            一个任意的字节串，通常是游戏状态
        """
        raise NotImplementedError

    async def tick(self, data: bytes) -> bytes:
        """
        执行一步游戏，由Client调用，在这里需要执行以下操作：
        1. 从data中解析出游戏状态
        2. 调用agent的remote_model.forward方法，获取action
        3. 将action序列化为字节串，返回给Client
        4. 收集trajectory，将其push到agent的remote_buffer中
        Args:
            data: 一个任意的字节串，由Client传入，通常是游戏状态
        Returns:
            一个任意的字节串，通常是序列化后的action
        """
        raise NotImplementedError

    async def stop(self, data: bytes) -> bytes:
        """
        游戏结束时调用，由Client调用，在这里需要执行以下操作：
        1. 从data中解析出游戏状态（通常是最后一帧的reward）
        2. 终止收集trajectory，将其push到agent的remote_buffer中
        3. 进行一些清理工作
        Args:
            data: 一个任意的字节串，由Client传入
        Returns:
            一个任意的字节串，通常是空的，或者一些统计信息
        """
        raise NotImplementedError

    def step(self, *args, **kwargs):
        """
        在这里实现tick输入序列化 -> tick调用 -> tick输出反序列化，
        这个函数的参数可以是任意格式，调用时保持一致即可
        Args:
            args: tick输入的位置参数
            kwargs: tick输入的关键字参数
        Returns:
            任意对象，为tick输出的反序列化后的值
        """
        raise NotImplementedError