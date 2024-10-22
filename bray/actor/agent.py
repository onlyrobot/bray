import asyncio
import os, sys
from typing import Any, Type
from google.protobuf.message import Message
import ray
from bray.actor.base import Actor
from bray.buffer.buffer import RemoteBuffer
from bray.master.master import register, get
from bray.metric.metric import get_step
import json
import pickle
import random


class State:
    """
    状态类，用于存储Actor的状态，可以通过.运算符或者wait方法修改和获取状态的属性，
    其中.运算符是同步的，wait方法是异步的，例如：
    # Agent1: `state.session = 1`
    # Agent2: `session = state.session`
    # Agent3: `session = await state.wait("session")`
    wait方法访问属性时，如果不存在，会等待属性被设置，从而实现Agent间状态同步
    """

    def __init__(self, session, actor, tick=0):
        self.conditions: dict[str : asyncio.Condition] = {}
        self.game, self.agent = session, ""
        self.session = session
        if len(parts := session.split("-")) >= 3:
            self.game = "".join(parts[:-2])
            self.agent = parts[-1]
        self.actor, self.tick = actor, tick

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        if not (cond := self.conditions.pop(__name, None)):
            return

        async def notify_all_get_attr(cond):
            async with cond:
                cond.notify_all()

        asyncio.create_task(notify_all_get_attr(cond))

    async def wait(self, __name: str) -> Any:
        if hasattr(self, __name):
            return self.__getattribute__(__name)

        if __name not in self.conditions:
            self.conditions[__name] = asyncio.Condition()

        async def wait_set_attr(coro) -> bool:
            try:
                await asyncio.wait_for(
                    coro,
                    timeout=10,
                )
                return True
            except asyncio.TimeoutError:
                return False

        async with (cond := self.conditions[__name]):
            retry = 3
            while retry and not await wait_set_attr(
                cond.wait_for(lambda: hasattr(self, __name))
            ):
                retry -= 1
                print(f"Wait {__name} timeout, retry...")
        return self.__getattribute__(__name)

    def __repr__(self) -> str:
        return f"<State {self.__dict__}>"


class Agent:
    def __init__(self, name: str, config: dict, state: State):
        """
        初始化一个新的Agent，当一局新的游戏开始时，会调用这个方法，
        你可以在这里初始化一些状态
        Args:
            name: Agent的名称，由Actor传入
            config: 全局配置，可以通过 config[name] 获取当前Agent的配置
            state: 当前session的状态，包括game, agent, session, actor属性
        """

    async def on_tick(self, state: State):
        """
        Actor每次tick都会调用这个方法，你需要在这里执行以下操作：
        1. 从state中获取本次tick的状态，例如 `input = await state.input`
        2. 执行当前Agent的决策逻辑，例如调用 `bray.forward` 方法
        3. 将输出放到state中，例如 `state.action = action`
        state中预设了一些属性，如下：
        state.session: 当前的游戏session，类型为str
        state.input: 当前tick的输入数据，类似为 bytes/proto/json
        state.output: 当前tick的输出数据，类似为 bytes/proto/json
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


def dump(state_path, state, session2path={}):
    if state.session not in session2path:
        path = os.path.join(
            state_path,
            f"step-{get_step()}-{state.session}",
        )
        session2path[state.session] = path
    path = session2path[state.session]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"tick-{state.tick}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)


@ray.remote(num_cpus=0, name="StateDumper")
def StateDumper(name: str, size: int):
    print("StateDumper started")
    trial_path = ray.get_runtime_context().namespace
    state_path = os.path.join(trial_path, "episode")
    if not os.path.exists(state_path):
        os.makedirs(state_path, exist_ok=True)
    print("Dump episode state to: ", state_path)

    state_buffer = RemoteBuffer(name, size=size)
    for state in state_buffer:
        try:
            dump(state_path, state)
        except Exception as e:
            print(e)
    print("StateDumper stopped")


def RemoteStateDumper(name="state", size=1024 * 4):
    from ray.util.scheduling_strategies import (
        NodeAffinitySchedulingStrategy,
    )

    scheduling_local = NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(),
        soft=False,
    )
    return StateDumper.options(
        scheduling_strategy=scheduling_local,
    ).remote(name, size=size)


class AgentActor(Actor):
    """
    AgentActor是一个特殊的的Actor，封装了Actor的start->tick->...->stop流程，
    以及数据序列化、反序列化的逻辑，提供了Agent、State和Episode的抽象，
    其中Agent代表游戏中的智能体，State代表了一次tick的状态，多个Agent可以共享状态，
    Episode代表了一段连续的tick，基于Episode可以计算出Agent的Replay用于训练
    """

    episode_offset: int = None

    def __init__(
        self,
        name: str = "default",
        Agents: dict[str, Type[Agent]] = {},
        episode_length: int = 128,
        episode_save_interval: int = None,
        serialize: str = None,
        TickInput: Type[Message] = None,
        TickOutput: Type[Message] = None,
    ):
        self.name, self.Agents = name, Agents
        self.config = get("config")
        self.actor_id = None
        self.episode_length = episode_length
        if episode_length is None:
            self.episode_length = sys.maxsize
        self.episode_save_interval = episode_save_interval
        if episode_save_interval is None:
            self.episode_save_interval = sys.maxsize
        AgentActor.episode_offset = random.randint(
            0,
            self.episode_save_interval,
        )
        self.state_buffer = None
        self.serialize = serialize
        self.TickInput = TickInput
        self.TickOutput = TickOutput

    async def start(self, session, _: bytes) -> bytes:
        if self.actor_id is None:
            self.actor_id = register(self.name)
        self.state_buffer = None
        if AgentActor.episode_offset % self.episode_save_interval == 0:
            self.state_buffer = RemoteBuffer("state")
        AgentActor.episode_offset += 1
        self.session = session
        self.tick_id = 0
        self.state = State(self.session, self.actor_id)
        self.agents: dict[str:Agent] = {
            name: Agent(
                name,
                self.config,
                self.state,
            )
            for name, Agent in self.Agents.items()
        }
        self.episode: list[State] = []
        return b"Game Started"

    async def tick(self, data: bytes) -> bytes:
        state = State(self.session, self.actor_id)
        state.input = data
        state.tick = self.tick_id
        self.tick_id += 1
        if self.serialize == "json":
            state.input = json.loads(data)
            state.output = {}
        elif self.serialize == "proto":
            state.input = self.TickInput()
            state.input.ParseFromString(data)
            state.output = self.TickOutput()
        await asyncio.gather(
            *[
                a.on_tick(
                    state,
                )
                for a in self.agents.values()
            ]
        )
        if len(self.episode) >= self.episode_length:
            tasks = [
                a.on_episode(
                    self.episode,
                    done=False,
                )
                for a in self.agents.values()
            ]
            asyncio.gather(*tasks)
            self.episode = []
        self.episode.append(state)
        if self.state_buffer:
            self.state_buffer.push(state)
        if isinstance(state.output, bytes):
            return state.output
        if self.serialize == "proto":
            return state.output.SerializeToString()
        return json.dumps(state.output).encode()

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
