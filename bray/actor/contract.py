from pydantic import BaseModel
from enum import Enum


class StepKind(str, Enum):
    start = "start"
    tick = "tick"
    end = "end"
    auto = "auto"


class StepRequest(BaseModel):
    game_id: str
    round_id: int
    kind: StepKind
    data: bytes


class StepRespone(BaseModel):
    game_id: str
    round_id: int
    kind: StepKind
    data: bytes