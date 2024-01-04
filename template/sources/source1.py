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

    def generate(self) -> Iterable[bray.NestedArray]:
        for _ in range(self.size):
            yield self.fake_data


def build_source(name, config: dict) -> list[Iterable[bray.NestedArray]]:
    """
    构建数据源，返回一个列表，列表中的每个元素都是一个可迭代数据集，
    返回数据集数量越多，生成数据的并行度越高
    Args:
        name: 数据源名称，在配置文件中指定
        config: 全局配置，通过 `config[name]` 获取当前数据源配置
    Returns:
        sources: 数据源列表，每个元素都是独立的数据集
    """
    return [AtariDataset() for _ in range(2)]


def build_eval_source(name, config: dict) -> list[Iterator[bray.NestedArray]]:
    return [AtariDataset() for _ in range(1)]


if __name__ == "__main__":
    import os, yaml

    path = os.path.dirname(__file__)
    with open(os.path.join(path, "../config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    source = build_source("train_source", config)
    print(source)
    print(next(iter(source[0])))
    eval_source = build_eval_source("eval_source", config)
    print(eval_source)
    print(next(iter(eval_source[0])))