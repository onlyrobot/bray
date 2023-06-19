import dataclasses

import bray

@dataclasses.dataclass
class Agent:
    remote_model: bray.RemoteModel
    remote_buffer: bray.RemoteBuffer