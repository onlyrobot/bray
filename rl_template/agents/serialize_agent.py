import numpy as np
import bray
import json
import base64


class SerializeAgent(bray.Agent):
    """
    负责输入输出的序列化和反序列化，具体的：
    1. 将state.input_data(bytes)反序列化为state.input(dict)
    2. 将state.output(dict)序列化为state.output_data(bytes)
    """

    async def on_tick(self, state: bray.State):
        input = json.loads(await state.input_data)
        obs = base64.b64decode(input["obs"])
        input["obs"] = np.frombuffer(
            obs,
            dtype=np.float32,
        ).reshape(42, 42, 4).transpose(2, 0, 1)
        state.input = input
        output = await state.output
        output_data = json.dumps(output).encode()
        state.output_data = output_data
