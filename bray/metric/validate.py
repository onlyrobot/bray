import numpy as np
from bray.metric.metric import merge


class Metric:
    def __init__(
        self,
        name: str = "metric",
        up_bound=float("inf"),
        low_bound=float("-inf"),
        max_samples=1000,
        print_report=False,
    ):
        """
        定义一个metric，用于统计某个指标的最大值、最小值、平均值、分位数等，
        支持numpy.ndarray，以及python的float和int类型，
        此外还可以指定指标的上下界，如果某个指标的值超出了上下界，会报错，用于值的校验
        Args:
            name: 指标的名称，用于打印和tensorboard输出
            up_bound: 指标的上界，用于值的校验
            low_bound: 指标的下界，用于值的校验
            max_samples: 采样间隔，每采样max_samples个数据，会打印一次指标的统计信息
            print_report: 是否在打印输出指标的统计信息
        """
        self.name = name
        self.cnt, self.sum = 0, 0.0
        self.up_bound = up_bound
        self.low_bound = low_bound
        self.max = float("-inf")
        self.min = float("inf")
        self.max_samples = max_samples
        self.samples = np.zeros(self.max_samples)
        self.num_samples = 0
        self.print_report = print_report

    def _merge(self, value: float | np.ndarray):
        if not isinstance(value, np.ndarray):
            self.samples[self.num_samples] = value
            self.num_samples += 1
            return None
        value = value.reshape(-1)
        merge_num = min(self.max_samples - self.num_samples, len(value))
        self.samples[self.num_samples : self.num_samples + merge_num] = value[
            :merge_num
        ]
        self.num_samples += merge_num
        return None if len(value) > merge_num else value[merge_num:]

    def merge(self, value: float | np.ndarray):
        assert np.all(self.low_bound <= value) and np.all(value <= self.up_bound), (
            f"Metric {self.name} value {value} out of bound [{self.low_bound}, "
            f"{self.up_bound}]"
        )
        self.cnt += 1
        self.sum += np.sum(value)
        self.max = max(self.max, np.max(value))
        self.min = min(self.min, np.max(value))
        remain_value = self._merge(value)
        if self.num_samples < self.max_samples:
            return
        self.report()
        self.num_samples = 0
        self.cnt, self.sum = 0, 0.0
        self.max = float("-inf")
        self.min = float("inf")
        if remain_value is None:
            return
        self._merge_ndarray_samples(remain_value)

    def report(self):
        samples = self.samples[: self.cnt]
        tiles = [(i, np.percentile(samples, i)) for i in [30, 50, 90, 95, 99]]
        merge(f"{self.name}/max", self.max)
        merge(f"{self.name}/min", self.min)
        merge(f"{self.name}/avg", self.avg)
        for i, tile in tiles:
            merge(f"{self.name}/{i}%", tile)
        if not self.print_report:
            return
        print(
            f"Metric {self.name} with {self.cnt} samples: "
            + f"max={self.max}, min={self.min}, "
            + f"avg={self.avg}, percentiles: {tiles}"
        )

    @property
    def avg(self):
        return self.sum / self.cnt


if __name__ == "__main__":
    metric = Metric()
    for i in range(10000):
        metric.merge(np.random.randn())
