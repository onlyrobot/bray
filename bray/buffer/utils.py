from typing import Iterator
import torch

from bray.utils.nested_array import NestedArray, make_batch


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
    def __init__(self, buffer: Iterator[NestedArray]):
        self.buffer = buffer

    def __next__(self) -> NestedArray:
        return torch.from_numpy(next(self.buffer))

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class TorchPrefetchBuffer:
    pass
