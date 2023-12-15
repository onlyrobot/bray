import bray
from google.protobuf.message import Message


async def on_start(global_state: bray.State):
    """
    当Actor的start方法被调用时，会调用所有agent的on_start方法，
    当不需要初始化时，可以不实现该方法
    Args:
        global_state: 对应一局游戏的全局状态，所有agent共享
    """
    global_state.step = 0
    print("agent on_start, game_id =", await global_state.game_id)


async def on_tick(
    global_state: bray.State,
    input: Message,
    output: Message,
    tick_state: bray.State,
):
    """
    当Actor的tick方法被调用时，会调用所有agent的on_tick方法
    Args:
        global_state: 对应一局游戏的全局状态，所有agent共享
        input: Actor的tick输入数据，为protobuf的Message类型
        output: Actor的tick返回数据，为protobuf的Message类型
        tick_state: 对应一次tick的状态
    """
    global_state.step = await global_state.step + 1
    print("agent on_tick, step =", await global_state.step)


async def on_episode(
    global_state: bray.State, episode: list[tuple[Message, Message, bray.State]]
):
    """
    当tick数量到达episode_length时，会调用所有agent的on_episode方法，
    在本方法中可以对episode数据进行处理，例如保存到文件中，push到Buffer中，
    当不需要处理episode数据时，可以不实现该方法
    Args:
        global_state: 对应一局游戏的全局状态，所有agent共享
        episode: 连续的episode数据，为list类型，每个元素为一个tuple，
            tuple中的三个值分别对应到tick方法的input, output, tick_state
    """
    print("agent on_episode, episode length =", len(episode))


async def on_end(global_state: bray.State):
    """
    当Actor的end方法被调用时，会调用所有agent的on_end方法，
    当不需要清理时，可以不实现该方法
    Args:
        global_state: 对应一局游戏的全局状态，所有agent共享
    """
    print("agent on_end")
