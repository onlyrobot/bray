from typing import Iterator
import torch
from threading import Thread, Condition

from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
)


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


class ReuseBuffer:
    pass


class TorchTensorBuffer:
    def __init__(self, buffer: Iterator[NestedArray], to_gpu: bool):
        self.buffer, self.to_gpu = buffer, to_gpu

    def handle(self, array):
        tensor = torch.from_numpy(array)
        if self.to_gpu:
            tensor = tensor.cuda()
        return tensor

    def __next__(self) -> NestedArray:
        return handle_nested_array(
            next(self.buffer),
            self.handle,
        )

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class TorchPrefetchBuffer:
    pass


class PrefetchBuffer:
    def __init__(self, buffer: Iterator[NestedArray]):
        self.buffer = buffer
        self.replays = []
        self.cond = Condition()
        self.prefetch_thread = Thread(target=self._thread)
        self.prefetch_thread.start()

    def _prefetch(self):
        with self.cond:
            self.cond.wait_for(lambda: len(self.replays) < 2)
            self.replays.append(next(self.buffer))
            self.cond.notify()

    def _thread(self):
        while True:
            self._prefetch()

    def __next__(self) -> NestedArray:
        with self.cond:
            self.cond.wait_for(lambda: len(self.replays) > 0)
            replay = self.replays.pop()
            self.cond.notify()
        return replay

    def __iter__(self) -> Iterator[NestedArray]:
        return self
