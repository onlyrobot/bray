from google.protobuf.message import Message
import uuid
import requests


class Client:
    def __init__(self, host: str, port: int = 8000):
        self.sess = requests.Session()
        self.url = f"http://{host}:{port}/step"
        self.game_id = ""

    def _request(self, step_kind, data):
        res = self.sess.post(
            url=self.url,
            headers={
                "game_id": self.game_id,
                "step_kind": step_kind,
            },
            data=data,
        )
        if res.status_code != 200:
            raise Exception(res.text)
        return res.content

    def start(self, game_id: str = None):
        """开始一局游戏，在结束游戏前，不要再次调用"""
        if game_id is None:
            game_id = str(uuid.uuid4())
        self.game_id = game_id
        self._request("start", b"")

    def tick(self, data: bytes) -> bytes:
        """游戏开始后，每一帧调用"""
        return self._request("tick", data)

    def stop(self):
        """结束当前游戏，准备开始下一局游戏"""
        self._request("stop", b"")

    def step(self, data: bytes) -> bytes:
        """无状态的tick接口，可以在任意时刻调用"""
        return self._request("step", data)


class AsyncClient:
    def __init__(self, host: str, port: int = 8000):
        import aiohttp

        self.sess = aiohttp.ClientSession()
        self.url = f"http://{host}:{port}/step"
        self.game_id = ""

    async def _request(self, step_kind, data):
        res = await self.sess.post(
            url=self.url,
            headers={
                "game_id": self.game_id,
                "step_kind": step_kind,
            },
            data=data,
        )
        if res.status != 200:
            raise Exception(await res.text())
        return await res.read()

    async def start(self, game_id: str = None):
        """开始一局游戏，在结束游戏前，不要再次调用"""
        if game_id is None:
            game_id = str(uuid.uuid4())
        self.game_id = game_id
        await self._request("start", b"")

    async def tick(self, data: bytes) -> bytes:
        """游戏开始后，每一帧调用"""
        return await self._request("tick", data)

    async def stop(self):
        """结束当前游戏，准备开始下一局游戏"""
        await self._request("stop", b"")

    async def step(self) -> bytes:
        """无状态的tick接口，可以在任意时刻调用"""
        return await self._request("step", b"")


class ProtobufClient(Client):
    def tick(self, input_msg: Message, output_msg: Message):
        """
        游戏开始后，每一帧调用，输入和输出都是Protobuf格式
        Args:
            input_msg: 输入的消息，会被自动序列化
            output_msg: 输出的消息，会被自动反序列化
        """
        data = input_msg.SerializeToString()
        output_msg.ParseFromString(super().tick(data))

    def step(self, input_msg: Message, output_msg: Message):
        """无状态的tick接口，可以在任意时刻调用"""
        data = input_msg.SerializeToString()
        output_msg.ParseFromString(super().tick(data))


class AsyncProtobufClient(AsyncClient):
    async def tick(self, input_msg: Message, output_msg: Message):
        """
        游戏开始后，每一帧调用，输入和输出都是Protobuf格式
        Args:
            input_msg: 输入的消息，会被自动序列化
            output_msg: 输出的消息，会被自动反序列化
        """
        data = input_msg.SerializeToString()
        output_msg.ParseFromString(await super().tick(data))

    async def step(self, input_msg: Message, output_msg: Message):
        """无状态的tick接口，可以在任意时刻调用"""
        data = input_msg.SerializeToString()
        output_msg.ParseFromString(await super().tick(data))
