import asyncio
import ray
import tensorboard
import time
import copy


class Metric:
    def __init__(self, cnt=0, sum=0.0):
        self.cnt = int(cnt)
        self.sum = float(sum)

    def merge(self, other: "Metric"):
        self.cnt += other.cnt
        self.sum += other.sum

    @property
    def avg(self):
        return self.sum / self.cnt


@ray.remote
class RemoteMetrics:
    async def __init__(self, time_window=60):
        self.time_window = time_window
        self.metrics, self.last_metrics = {}, {}
        self.writer = tensorboard.summary.Writer("./metrics/tensorboard")
        self.beg = time.time()
        asyncio.create_task(self.dump_to_tensorboard())

    def merge(self, name, metric: Metric):
        m = self.metrics.get(name)
        if not m:
            self.metrics[name] = metric
        else:
            m.merge(metric)

    def query(self, name):
        return self.metrics.get(name, Metric())

    async def dump_to_tensorboard(self):
        await asyncio.sleep(self.time_window)
        step = (time.time() - self.beg) // self.time_window
        current_metrics = copy.deepcopy(self.metrics)
        for name, current in current_metrics.items():
            last = self.last_metrics.get(name, Metric())
            diff = Metric(
                current.cnt - last.cnt,
                current.sum - last.sum,
            )
            if diff.cnt == 0:
                continue
            self.writer.add_scalar(name, diff.avg, step)
        self.writer.flush()
        self.last_metrics = current_metrics
        asyncio.create_task(self.dump_to_tensorboard())


remote_metrics = RemoteMetrics.options(
    name="RemoteMetrics", get_if_exists=True, lifetime="detached"
).remote()


def merge(name: str, value: float) -> ray.ObjectRef:
    return remote_metrics.merge.remote(name, Metric(1, value))


def query(name: str, kind: str = "avg") -> float | int:
    """
    Args:
        name: metric name
        kind: avg, sum, cnt
    """
    metric = ray.get(remote_metrics.query.remote(name))
    if kind == "avg":
        return float("nan") if metric.cnt == 0 else metric.avg
    elif kind == "sum":
        return metric.sum
    elif kind == "cnt":
        return metric.cnt
    else:
        raise ValueError(f"Unsupported kind: {kind}")


if __name__ == "__main__":
    ray.get(merge("test", 1))
    assert query("test") == 1
    ray.get(merge("test", 2))
    assert query("test") == 1.5
    merge("test", 3)
    time.sleep(5)
    assert query("test") == 2
    print("pass")