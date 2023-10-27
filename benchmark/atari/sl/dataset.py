from typing import Iterator
import bray


class AtariDataset:
    def __init__(self, fake_data: bray.NestedArray, size=1000):
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
