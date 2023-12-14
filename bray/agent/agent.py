import asyncio
from bray.actor.base import Actor
from types import ModuleType
from typing import Any, Type
from google.protobuf.message import Message


class State:
    """
    状态类，用于存储Actor的状态，可以通过.运算符访问和修改状态的属性，
    其中访问属性是异步的，修改属性是同步的，例如：
    ```
    # Agent1
    state.game_id = 1

    # Agent2
    await state.game_id
    ```
    这样设计是为了在多个Agent共享状态时，可以通过await来同步执行顺序
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
                    timeout=5,
                )
                return True
            except asyncio.TimeoutError:
                return False

        async with (cond := conditions[__name]):
            while not await wait_for_set_attr(
                cond.wait_for(lambda: _hasattr(self, __name))
            ):
                print(f"Get attr {__name} timeout, retry...")
        return self.__getattribute__(__name)


class AgentActor(Actor):
    """
    AgentActor继承自Actor，是一个Actor的实现，用于将多个Agent组合在一起，
    当Gamecore调用AgentActor的start、tick、end方法时，会依次调用
    所有Agent的on_start、on_tick、on_end方法，当收集到的episode长度达到
    episode_length时，会调用所有Agent的on_episode方法
    """

    def __init__(
        self,
        TickInputProto: Type[Message],
        TickOutputProto: Type[Message],
        agents: dict[str:ModuleType],
        episode_length: int = 128,
    ):
        """ """
        self.TickInputProto = TickInputProto
        self.TickOutputProto = TickOutputProto
        self.agents = agents
        self.episode_length = episode_length

    async def start(self, game_id, data: bytes) -> bytes:
        self.global_state = State()
        self.global_state.game_id = game_id
        await asyncio.gather(
            *[
                a.on_start(
                    self.global_state,
                )
                for a in self.agents
            ]
        )
        self.episode = []
        return b"Game Started"

    async def tick(self, data: bytes) -> bytes:
        input = self.TickInputProto()
        input.ParseFromString(data)
        output = self.TickOutputProto()
        state = State()
        self.episode.append([input, output, state])
        await asyncio.gather(
            *[
                a.on_tick(
                    self.global_state,
                    input,
                    output,
                    state,
                )
                for a in self.agents
            ]
        )
        if len(self.episode) < self.episode_length:
            return output.SerializeToString()
        asyncio.gather(
            *[
                a.on_episode(
                    self.global_state,
                    self.episode,
                )
                for a in self.agents
            ]
        )
        self.episode = []
        return output.SerializeToString()

    async def end(self, data: bytes) -> bytes:
        await asyncio.gather(
            *[
                a.on_end(
                    self.global_state,
                )
                for a in self.agents
            ]
        )
        del self.global_state, self.episode
        return b"Game Ended"


if __name__ == "__main__":
    import importlib

    agent_module = importlib.import_module("bray.agent.base")
    agent_module2 = importlib.import_module("bray.agent.base")

    class FakeProto:
        def SerializeToString(self):
            return b""

        def ParseFromString(self, data):
            pass

    agent_actor = AgentActor(
        FakeProto,
        FakeProto,
        [agent_module, agent_module2],
    )

    async def test():
        await agent_actor.start(1, b"")
        for _ in range(256):
            print(await agent_actor.tick(b""))
        await agent_actor.end(b"")

    asyncio.run(test())
