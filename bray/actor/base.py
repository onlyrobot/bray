class Actor:
    """
    Actor是一个有状态服务接受来自Gamecore的step调用，调用的顺序是：
    start -> tick -> tick -> ... -> end
    """

    def __init__(self, *args, **kwargs):
        """
        初始化一个新的Actor，当一局新的游戏开始之前时，会调用这个方法
        Args:
            *args: 位置参数，由RemoteActor传入
            **kwargs: 关键字参数，由RemoteActor传入
        """
        raise NotImplementedError

    def start(self, game_id, data: bytes) -> bytes:
        """
        开始一局新的游戏，由Gamecore调用，请在这里初始化游戏状态
        Args:
            game_id: 一个唯一的游戏ID，由Gamecore传入
            data: 一个任意的字节串，由Gamecore传入，通常是游戏状态
        Returns:
            一个任意的字节串，通常是游戏状态
        """
        raise NotImplementedError

    async def tick(self, data: bytes) -> bytes:
        """
        执行一步游戏，由Gamecore调用，在这里需要执行以下操作：
        1. 从data中解析出游戏状态
        2. 调用agent的remote_model.forward方法，获取action
        3. 将action序列化为字节串，返回给Gamecore
        4. 收集trajectory，将其push到agent的remote_buffer中
        Args:
            data: 一个任意的字节串，由Gamecore传入，通常是游戏状态
        Returns:
            一个任意的字节串，通常是序列化后的action
        """
        raise NotImplementedError

    def end(self, data: bytes) -> bytes:
        """
        游戏结束时调用，由Gamecore调用，在这里需要执行以下操作：
        1. 从data中解析出游戏状态（通常是最后一帧的reward）
        2. 终止收集trajectory，将其push到agent的remote_buffer中
        3. 进行一些清理工作
        Args:
            data: 一个任意的字节串，由Gamecore传入
        Returns:
            一个任意的字节串，通常是空的，或者一些统计信息
        """
        raise NotImplementedError
