import asyncio
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray import get_runtime_context

GLOBAL_EVENT_LOOP: asyncio.BaseEventLoop = None

def create_or_get_event_loop() -> asyncio.BaseEventLoop:
    global GLOBAL_EVENT_LOOP
    if GLOBAL_EVENT_LOOP:
        return GLOBAL_EVENT_LOOP
    while GLOBAL_EVENT_LOOP is False: pass
    if GLOBAL_EVENT_LOOP: return GLOBAL_EVENT_LOOP
    GLOBAL_EVENT_LOOP = False
    GLOBAL_EVENT_LOOP = asyncio.new_event_loop()
    from threading import Thread

    Thread(target=GLOBAL_EVENT_LOOP.run_forever, daemon=True
    ).start()
    return GLOBAL_EVENT_LOOP

def ray_scheduling_local(node_id: str = None):
    if not node_id:
        node_id = get_runtime_context().get_node_id()
    return NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False)
