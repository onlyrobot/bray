import asyncio
from typing import Any, Type
from google.protobuf.message import Message
from bray.actor.base import Actor
from bray.master.master import register


class State:
    """
    状态类，用于存储Actor的状态，可以通过.运算符访问和设置状态的属性，
    其中访问属性是异步的，设置属性是同步的，例如：
    # Agent1: `state.game_id = 1`
    # Agent2: `game_id = await state.game_id`
    访问属性时，如果不存在，会等待属性被设置，从而实现Agent间状态同步
    """

    def __init__(self):
        self.conditions: dict[str : asyncio.Condition] = {}

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        conditions = super().__getattribute__("conditions")
        if not (cond := conditions.pop(__name, None)):
            return

        async def notify_all_get_attr(cond):
            async with cond:
                cond.notify_all()

        asyncio.create_task(notify_all_get_attr(cond))

    def get(self, __name: str, __default: Any = None) -> Any:
        """
        获取状态的属性，如果属性不存在，则返回默认值
        """
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return __default

    async def __getattribute__(self, __name: str) -> Any:
        def _hasattr(__obj, __name) -> bool:
            try:
                __obj.__getattribute__(__name)
                return True
            except AttributeError:
                return False

        if _hasattr(self := super(), __name):
            return self.__getattribute__(__name)
        conditions = self.__getattribute__("conditions")
        if __name not in conditions:
            conditions[__name] = asyncio.Condition()

        async def wait_for_set_attr(coro) -> bool:
            try:
                await asyncio.wait_for(
                    coro,
                    timeout=10,
                )
                return True
            except asyncio.TimeoutError:
                return False

        async with (cond := conditions[__name]):
            retry = 3
            while (retry := retry - 1) and not await wait_for_set_attr(
                cond.wait_for(lambda: _hasattr(self, __name))
            ):
                print(f"Get attr {__name} timeout, retry...")
        return self.__getattribute__(__name)


class Agent:
    def __init__(self, name: str):
        """
        初始化一个新的Agent，当一局新的游戏开始时，会调用这个方法，
        你可以在这里初始化一些状态
        Args:
            name: Agent的名称，由Actor传入
        """

    async def on_tick(self, state: State):
        """
        Actor每次tick都会调用这个方法，你需要在这里执行以下操作：
        1. 从state中获取本次tick的状态，例如 `input = await state.input`
        2. 执行当前Agent的决策逻辑，例如调用 `bray.forward` 方法
        3. 将输出放到state中，例如 `state.action = action`
        state中预设了一些属性，如下：
        state.game_id: 当前游戏的ID
        state.data: 当前tick的原始输入数据，类型为bytes
        state.input: 当前tick反序列化后的输入数据，类型为Protobuf的Message
        state.output: 当前tick的输出数据，类型为Protobuf的Message
        Args:
            state: 当前tick的状态，可以通过.运算符访问和设置状态的属性
        """
        raise NotImplementedError

    async def on_episode(self, episode: list[State], done: bool):
        """
        当收集到的episode长度达到episode_length或者游戏结束时，
        会调用这个方法，你可以在这里执行以下操作：
        1. 从episode中获取当前episode的状态
        2. 根据episode的状态，构建trajectory并计算reward
        3. 将trajectory推送到缓冲区，例如 `bray.push("buffer1", *trajectory)`
        Args:
            episode: 一个episode，包含了连续的多个tick的state
            done: 是否为最后一个episode
        """


class AgentActor(Actor):
    def __init__(
        self,
        name: str = "default",
        Agents: dict[str, Type[Agent]] = {},
        episode_length: int = 128,
        TickInputProto: Type[Message] = None,
        TickOutputProto: Type[Message] = None,
    ):
        self.name, self.Agents = name, Agents
        self.actor_id = register(self.name)
        self.episode_length = episode_length
        self.TickInputProto = TickInputProto
        self.TickOutputProto = TickOutputProto

    async def start(self, game_id, _: bytes) -> bytes:
        self.game_id = game_id
        self.agents: dict[str:Agent] = {
            name: Agent(
                name,
            )
            for name, Agent in self.Agents.items()
        }
        self.episode: list[State] = []
        return b"Game Started"

    async def tick(self, data: bytes) -> bytes:
        state = State()
        state.actor_id = self.actor_id
        state.game_id = self.game_id
        state.input_data = data
        if self.TickInputProto:
            input = self.TickInputProto()
            input.ParseFromString(data)
            state.input = input
            state.output = self.TickOutputProto()
        self.episode.append(state)
        await asyncio.gather(
            *[
                a.on_tick(
                    state,
                )
                for a in self.agents.values()
            ]
        )
        if self.TickInputProto:
            output = await state.output
            state.output_data = output.SerializeToString()
        if (
            not self.episode_length
            or len(
                self.episode,
            )
            < self.episode_length
        ):
            return await state.output_data
        asyncio.gather(
            *[
                a.on_episode(
                    self.episode,
                    done=False,
                )
                for a in self.agents.values()
            ]
        )
        self.episode = []
        return await state.output_data

    async def stop(self, _: bytes) -> bytes:
        asyncio.gather(
            *[
                a.on_episode(
                    self.episode,
                    done=True,
                )
                for a in self.agents.values()
            ]
        )
        self.episode = []
        return b"Game Stopped"
