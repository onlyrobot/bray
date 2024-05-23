from typing import Iterator, Callable, List
import torch
import numpy as np
from threading import Thread, Condition

from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
)

from bray.master.master import merge


class ListBuffer:
    def __init__(self, buffer: Iterator[NestedArray], size):
        self.buffer, self.size = buffer, size

    def __next__(self) -> NestedArray:
        return [next(self.buffer) for _ in range(self.size)]

    def __iter__(self) -> Iterator[NestedArray]:
        return self


def to_paged_memory(nested_array: NestedArray):
    return handle_nested_array(
        nested_array, 
        lambda x: torch.as_tensor(x).pin_memory().numpy())


class StackBuffer:
    def __init__(self, buffer: Iterator[NestedArray], size):
        self.last_batch = None
        self.buffer = ListBuffer(buffer, size)

    def __next__(self) -> NestedArray:
        batch = make_batch(
            next(self.buffer), out=self.last_batch
        )
        if self.last_batch is None:
            self.last_batch = (to_paged_memory(batch) if 
                torch.cuda.is_available() else batch)
        self.last_batch = batch
        return self.last_batch

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class ConcateBuffer:
    def __init__(self, buffer: Iterator[NestedArray], size):
        self.last_batch = None
        self.buffer = ListBuffer(buffer, size)

    def __next__(self) -> NestedArray:
        batch = make_batch(
            next(self.buffer), concate=True, out=self.last_batch
        )
        if self.last_batch is None:
            self.last_batch = (to_paged_memory(batch) if 
                torch.cuda.is_available() else batch)
        return self.last_batch

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class TorchTensorBuffer:
    def __init__(
        self, buffer: Iterator[NestedArray], device=None
    ):
        self.buffer, self.device = buffer, device

    def handle(self, array):
        tensor = torch.as_tensor(array)
        if self.device:
            tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    def __next__(self) -> NestedArray:
        return handle_nested_array(next(self.buffer), self.handle)

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class CallbackBuffer:
    def __init__(
        self, buffer: Iterator[NestedArray],
        callback: Callable[[NestedArray], NestedArray],
    ):
        self.buffer, self.callback = buffer, callback

    def __next__(self) -> NestedArray:
        return self.callback(next(self.buffer))

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class PrefetchBuffer:
    is_reuse: bool = False
    def __init__(self, buffer: Iterator[NestedArray], size=1, max_reuse=0, name=""):
        """
        Args:
            size: 缓冲区大小，即预取的样本数量，最小为1
            max_reuse: 样本的最大重用次数，设为0关闭重用
            name: buffer名称，用于reuse指标的统计
        """
        assert size >= 1, "Prefetch buffer size must be greater than 1"
        self.buffer, self.size, self.name = buffer, size, name
        self.max_reuse, self.remain_reuse = max_reuse, max_reuse
        self.last_reuse = 0
        self.replays, self.last_replay = [], None
        self.cond = Condition()
        self.prefetch_thread = Thread(target=self._thread)
        self.prefetch_thread.start()

    def _prefetch(self):
        with self.cond:
            self.cond.wait_for(lambda: len(self.replays) < self.size)
            self.replays.append(next(self.buffer))
            self.cond.notify()
        if self.max_reuse > 0 and self.name:
            merge(f"reuse/{self.name}", self.last_reuse)

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
            PrefetchBuffer.is_reuse = True
            return self.last_replay
        with self.cond:
            self.cond.wait_for(lambda: len(self.replays) > 0)
            self.last_replay = self.replays.pop()
            self.last_reuse = self.max_reuse - self.remain_reuse
            self.remain_reuse = self.max_reuse
            self.cond.notify()
        PrefetchBuffer.is_reuse = False
        return self.last_replay

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class SampleBuffer:
    def __init__(self, buffers: List[Iterator], weights=None):
        """
        Args:
            buffers: 迭代器列表
            weights: 每个迭代器的权重，如果为None则默认为均匀分布
        """
        self.buffers = buffers
        if not weights:
            weights = np.array([1.0] * len(buffers))
        self.weights = np.array(weights) / np.sum(weights)

    def __next__(self):
        return next(np.random.choice(self.buffers, p=self.weights))

    def __iter__(self) -> Iterator[NestedArray]:
        return self
