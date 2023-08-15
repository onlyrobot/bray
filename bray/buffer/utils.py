from typing import Iterator
import torch
from threading import Thread, Condition

from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
)

from bray.metric.metric import merge


class BatchBuffer:
    def __init__(self, buffer: Iterator[NestedArray], batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __next__(self) -> NestedArray:
        batch = []
        for _ in range(self.batch_size):
            batch.append(next(self.buffer))
        return make_batch(batch)

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class TorchTensorBuffer:
    def __init__(self, buffer: Iterator[NestedArray], device: None):
        self.buffer, self.device = buffer, device

    def handle(self, array):
        tensor = torch.from_numpy(array)
        if self.device:
            tensor = tensor.to(self.device)
        return tensor

    def __next__(self) -> NestedArray:
        return handle_nested_array(next(self.buffer), self.handle)

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class CallbackBuffer:
    def __init__(self, buffer: Iterator[NestedArray], callback):
        self.buffer, self.callback = buffer, callback

    def __next__(self) -> NestedArray:
        return self.callback(next(self.buffer))

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class TorchPrefetchBuffer:
    pass


class PrefetchBuffer:
    def __init__(
        self, buffer: Iterator[NestedArray], max_reuse: int = 0, name: str = "default"
    ):
        """
        Args:
            buffer: 迭代器
            max_reuse: 样本的最大重用次数，设为0关闭重用
            name: buffer名称，用于reuse指标的统计
        """
        self.buffer, self.name = buffer, name
        self.max_reuse, self.remain_reuse = max_reuse, max_reuse
        self.last_reuse = 0
        self.replays, self.last_replay = [], None
        self.cond = Condition()
        self.prefetch_thread = Thread(target=self._thread)
        self.prefetch_thread.start()

    def _prefetch(self):
        with self.cond:
            self.cond.wait_for(lambda: len(self.replays) < 1)
            self.replays.append(next(self.buffer))
            self.cond.notify()
        merge("reuse", self.last_reuse, buffer=self.name)

    def _thread(self):
        while True:
            self._prefetch()

    def __next__(self) -> NestedArray:
        if (
            len(self.replays) == 0
            and self.last_replay is not None
            and self.remain_reuse > 0
        ):
            self.remain_reuse -= 1
            return self.last_replay
        with self.cond:
            self.cond.wait_for(lambda: len(self.replays) > 0)
            self.last_replay = self.replays.pop()
            self.last_reuse = self.max_reuse - self.remain_reuse
            self.remain_reuse = self.max_reuse
            self.cond.notify()
        return self.last_replay

    def __iter__(self) -> Iterator[NestedArray]:
        return self
