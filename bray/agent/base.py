import bray
from google.protobuf.message import Message


async def on_start(global_state: bray.State):
    global_state.step = 0
    print("agent on_start, game_id =", await global_state.game_id)


async def on_tick(
    global_state: bray.State,
    input: Message,
    output: Message,
    tick_state: bray.State,
):
    global_state.step = await global_state.step + 1
    print("agent on_tick, step =", await global_state.step)


async def on_episode(global_state: bray.State, episode):
    print("agent on_episode, episode length =", len(episode))


async def on_end(global_state: bray.State):
    print("agent on_end")
