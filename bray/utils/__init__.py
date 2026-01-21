import asyncio, threading

GLOBAL_EVENT_LOOP: asyncio.BaseEventLoop = None

def create_or_get_event_loop() -> asyncio.BaseEventLoop:
    global GLOBAL_EVENT_LOOP
    if GLOBAL_EVENT_LOOP: return GLOBAL_EVENT_LOOP
    while GLOBAL_EVENT_LOOP is False: pass
    if GLOBAL_EVENT_LOOP: return GLOBAL_EVENT_LOOP
    GLOBAL_EVENT_LOOP = False
    GLOBAL_EVENT_LOOP = asyncio.new_event_loop()

    threading.Thread(target=GLOBAL_EVENT_LOOP.run_forever, 
    daemon=True).start()
    return GLOBAL_EVENT_LOOP