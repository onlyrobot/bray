import numpy as np


class Metric:
    def __init__(
        self,
        name: str = "metric",
        max=float("inf"),
        min=float("-inf"),
        max_samples=1000,
    ):
        self.name = name
        self.cnt, self.sum = 0, 0.0
        self.up_bound = max
        self.low_bound = min
        self.max = float("-inf")
        self.min = float("inf")
        self.max_samples = max_samples
        self.samples = np.zeros(self.max_samples)
        self.num_samples = 0

    def merge(self, value: float):
        assert self.low_bound <= value <= self.up_bound, (
            f"Metric {self.name} value {value} out of bound [{self.low_bound}, "
            f"{self.up_bound}]"
        )
        self.cnt += 1
        self.sum += value
        self.max = max(self.max, value)
        self.min = min(self.min, value)
        self.samples[self.num_samples % self.max_samples] = value
        self.num_samples += 1
        if self.num_samples % self.max_samples != 0:
            return
        self.report()
        self.num_samples = 0
        self.cnt, self.sum = 0, 0.0
        self.max = float("-inf")
        self.min = float("inf")

    def report(self):
        samples = self.samples[: self.cnt]
        print(
            f"Metric {self.name} with {self.cnt} samples: "
            + f"max={self.max}, min={self.min}, avg={self.avg},"
            + f"30% percentil={np.percentile(samples, 30)}, "
            + f"50% percentil={np.percentile(samples, 50)}, "
            + f"90% percentil={np.percentile(samples, 90)}, "
            + f"95% percentil={np.percentile(samples, 95)}, "
            + f"99% percentil={np.percentile(samples, 99)}, "
        )

    @property
    def avg(self):
        return self.sum / self.cnt


if __name__ == "__main__":
    metric = Metric()
    for i in range(10000):
        metric.merge(np.random.randn())
