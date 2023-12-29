from google.protobuf.message import Message
import uuid
import requests


class Actor:
    """移动端部署时的Actor对象，运行的是环境和AI的交互逻辑"""

    def __init__(self):
        """构造函数，初始化状态、模型等"""
        raise NotImplementedError

    def start(self, game_id: str):
        """开始一局游戏，在结束游戏前，不要再次调用"""
        raise NotImplementedError

    def tick(self, input: Message, output: Message):
        """
        游戏开始后，每一帧调用。
        输入和输出都是Protobuf的Message，这里需要实现AI的逻辑，
        解析input，调用模型，填充output
        """
        raise NotImplementedError

    def stop(self):
        """结束当前游戏，准备开始下一局游戏"""
        raise NotImplementedError


class Client:
    def __init__(
        self,
        host: str,
        port: int = 8000,
        timeout: int = 60,
        actor: Actor = None,
        key: str = None,
        secret: str = None,
        token: str = None,
    ):
        """
        AI服务的客户端，当actor为None时，为服务端部署，否则为移动端部署
        Args:
            host: AI服务的主机名，可以是域名或者IP地址
            port: 服务端的端口号
            timeout: Client接口的超时时间
            actor: 移动端部署时的Actor对象
            key: API Key
            secret: API Secret
            token: API Token
        """
        self.timeout = timeout
        self.actor = actor
        self.key, self.secret, self.token = key, secret, token
        self.sess = None
        self.url = f"http://{host}:{port}/step"
        self.game_id = ""

    def _request(self, step_kind, data):
        if self.sess is None:
            self.sess = requests.Session()
        res = self.sess.post(
            url=self.url,
            headers={
                "game_id": self.game_id,
                "step_kind": step_kind,
            },
            data=data,
            timeout=self.timeout,
        )
        if res.status_code != 200:
            raise Exception(res.text)
        return res.content

    def start(self, game_id: str = None):
        """开始一局游戏，在上一局游戏结束前，不要再次调用"""
        if game_id is None:
            game_id = str(uuid.uuid4())
        self.game_id = game_id
        if self.actor is None:
            self._request("start", b"")
        else:
            self.actor.start(game_id)

    def tick(self, input: Message, output: Message):
        """
        游戏开始后，每一帧调用。
        输入和输出都是Protobuf的Message，会自动序列化和反序列化
        """
        if self.actor is None:
            data = input.SerializeToString()
            output.ParseFromString(self._request("tick", data))
        else:
            self.actor.tick(input, output)

    def _tick(self, data: bytes) -> bytes:
        """直接将序列化后的数据发送给服务端，返回序列化后的数据"""
        return self._request("tick", data)

    def stop(self):
        """结束当前游戏，准备开始下一局游戏"""
        if self.actor is None:
            self._request("stop", b"")
        else:
            self.actor.stop()

    def step(self, input: Message, output: Message):
        """无状态的tick接口，可以在任意时刻调用"""
        if self.actor is None:
            data = input.SerializeToString()
            output.ParseFromString(self._request("tick", data))
        else:
            self.actor.tick(input, output)
