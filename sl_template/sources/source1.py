from typing import Iterator, Iterable
import bray
import numpy as np


class AtariDataset(Iterable[bray.NestedArray]):
    def __init__(self, size=10000):
        self.fake_data = {
            "state": {"image": np.random.randn(4, 42, 42).astype(np.float32)},
            "action": np.random.randint(0, 9),
        }
        self.size = size

    def __iter__(self) -> Iterator[bray.NestedArray]:
        """
        返回一个迭代器，本方法可能会被多次调用，每次调用都会返回一个新的迭代器。
        """
        return self.generate()

    def generate(self) -> Iterator[bray.NestedArray]:
        for _ in range(self.size):
            yield self.fake_data


def build_source() -> list[Iterator[bray.NestedArray]]:
    return [AtariDataset() for _ in range(10)]


def build_eval_source() -> list[Iterator[bray.NestedArray]]:
    return [AtariDataset() for _ in range(2)]