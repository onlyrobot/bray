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
        trial_path = ray.get_runtime_context().namespace
        self.writer = tensorboard.summary.Writer(f"{trial_path}/tensorboard")
        self.beg = time.time()
        asyncio.create_task(self.dump_to_tensorboard())

    def _build_name(self, name, **kwargs) -> str:
        if not kwargs:
            return name
        attributes = [f"{k}={v}" for k, v in kwargs.items()]
        return f"{name}{{{', '.join(attributes)}}}"

    def merge(self, name, metric: Metric, **kwargs):
        name = self._build_name(name, **kwargs)
        m = self.metrics.get(name)
        if m:
            m.merge(metric)
        else:
            self.metrics[name] = metric

    def query(self, name, **kwargs) -> Metric:
        name = self._build_name(name, **kwargs)
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
        # self.writer.flush()
        self.last_metrics = current_metrics
        asyncio.create_task(self.dump_to_tensorboard())


class MetricsWorker:
    def __init__(self):
        self.remote_metrics = RemoteMetrics.options(
            name="RemoteMetrics", get_if_exists=True, lifetime="detached"
        ).remote()

    def merge(self, name: str, metric: Metric, **kwargs):
        self.remote_metrics.merge.remote(name, metric, **kwargs)

    def query(self, name: str, **kwargs) -> Metric:
        return ray.get(self.remote_metrics.query.remote(name, **kwargs))


metrics_worker = None


def get_metrics_worker() -> MetricsWorker:
    global metrics_worker
    if metrics_worker is None:
        metrics_worker = MetricsWorker()
    return metrics_worker


def merge(name: str, value: float, **kwargs):
    """
    输出指标到TensorBoard，支持在集群任何地方调用
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        value: 指标的值
    """
    metrics_worker = get_metrics_worker()
    metrics_worker.merge(name, Metric(1, value), **kwargs)


def query(name: str, kind: str = "avg", **kwargs) -> float | int:
    """
    查询指标，支持在集群任何地方调用
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        kind: 可以为 avg, sum, cnt
    """
    metrics_worker = get_metrics_worker()
    metric = metrics_worker.query(name, **kwargs)
    if kind == "avg":
        return float("nan") if metric.cnt == 0 else metric.avg
    elif kind == "sum":
        return metric.sum
    elif kind == "cnt":
        return metric.cnt
    else:
        raise ValueError(f"Unsupported kind: {kind}")