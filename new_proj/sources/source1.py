from typing import Iterator
import bray
import numpy as np


class AtariDataset:
    def __init__(self, fake_data: bray.NestedArray, size=10):
        self.fake_data = fake_data
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
    """
    返回一个迭代器列表，每个迭代器的元素为嵌套的np.ndarray类型
    """
    fake_data = {
        "input": np.random.randn(42, 42, 4).astype(np.float32),
        "label": np.array(0.0, dtype=np.float32),
    }
    return [AtariDataset(fake_data) for _ in range(10)]