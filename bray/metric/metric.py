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
        self.diff_metrics = {}
        self.descs = {}
        trial_path = ray.get_runtime_context().namespace
        self.writer = tensorboard.summary.Writer(f"{trial_path}/tensorboard")
        self.step = 0
        asyncio.create_task(self.dump_to_tensorboard())

    def _diff(self, current: Metric, last: Metric) -> Metric:
        return Metric(current.cnt - last.cnt, current.sum - last.sum)

    def merge(self, name, metric, desc: dict[str:str]):
        if desc:
            self.descs[name] = desc
        m = self.metrics.get(name, None)
        if m:
            m.merge(metric)
        else:
            self.metrics[name] = metric

    def query(self, name, time_window: bool) -> Metric:
        if not time_window:
            return self.metrics.get(name, Metric())
        return self.diff_metrics.get(name, Metric())

    def _dump_by_desc(self, name, metric, diff):
        desc = self.descs.get(name, None)
        if not desc:
            if diff.cnt != 0:
                self.writer.add_scalar(name, diff.avg, self.step)
            return
        if metric.cnt != 0:
            if "avg" in desc and metric.cnt != 0:
                self.writer.add_scalar(
                    f"{name} -- {desc['avg']}",
                    metric.avg,
                    self.step,
                )
            if "sum" in desc:
                self.writer.add_scalar(
                    f"{name} -- {desc['sum']}",
                    metric.sum,
                    self.step,
                )
            if "cnt" in desc:
                self.writer.add_scalar(
                    f"{name} -- {desc['cnt']}",
                    metric.cnt,
                    self.step,
                )
        if diff.cnt != 0:
            if "time_window_avg" in desc and diff.cnt != 0:
                self.writer.add_scalar(
                    f"{name} -- {desc['time_window_avg']}",
                    diff.avg,
                    self.step,
                )
            if "time_window_sum" in desc:
                self.writer.add_scalar(
                    f"{name} -- {desc['time_window_sum']}",
                    diff.sum,
                    self.step,
                )
            if "time_window_cnt" in desc:
                self.writer.add_scalar(
                    f"{name} -- {desc['time_window_cnt']}",
                    diff.cnt,
                    self.step,
                )

    async def dump_to_tensorboard(self):
        await asyncio.sleep(self.time_window)
        self.step += 1
        current_metrics = copy.deepcopy(self.metrics)
        for name, current in current_metrics.items():
            last = self.last_metrics.get(name, Metric())
            diff = self._diff(current, last)
            # diff metrics used for time window query
            self.diff_metrics[name] = diff
            self._dump_by_desc(name, current, diff)
        self.last_metrics = current_metrics
        asyncio.create_task(self.dump_to_tensorboard())


def build_name(name, **kwargs) -> str:
    if not kwargs:
        return name
    attributes = [f"{k}={v}" for k, v in kwargs.items()]
    return f"{name} {{{', '.join(attributes)}}}"


class MetricsWorker:
    def __init__(self):
        self.remote_metrics = RemoteMetrics.options(
            name="RemoteMetrics", get_if_exists=True, lifetime="detached"
        ).remote()
        self.cached_metrics = {}
        self.last_merge_remote_time = time.time()
        self.merge_remote_interval = 1
        self.merge_count = 0

    def merge_to_remote(self):
        for name, m in self.cached_metrics.items():
            self.remote_metrics.merge.remote(name, m, None)
        self.cached_metrics = {}

    def merge(self, name: str, metric: Metric, desc: dict[str:str], **kwargs):
        self.merge_count += 1
        name = build_name(name, **kwargs)
        m = self.cached_metrics.get(name, None)
        if not m:
            self.remote_metrics.merge.remote(name, metric, desc)
            self.cached_metrics[name] = Metric()
        else:
            m.merge(metric)
        if self.merge_count < self.merge_remote_interval:
            return
        self.merge_to_remote()
        merge_remote_time = time.time()
        self.merge_remote_interval = int(
            2
            + (self.merge_count - 1)
            * 60
            / (merge_remote_time - self.last_merge_remote_time)
        )
        self.last_merge_remote_time = merge_remote_time
        self.merge_count = 0

    def query(self, name, time_window, **kwargs) -> Metric:
        name = build_name(name, **kwargs)
        return ray.get(self.remote_metrics.query.remote(name, time_window))

    def __del__(self):
        self.merge_to_remote()


metrics_worker = None


def get_metrics_worker() -> MetricsWorker:
    global metrics_worker
    if metrics_worker is None:
        metrics_worker = MetricsWorker()
    return metrics_worker


def flush_metrics_to_remote():
    global metrics_worker
    if not metrics_worker:
        return
    metrics_worker.merge_to_remote()


def merge(name: str, value: float, desc: dict[str:str] = None, **kwargs):
    """
    输出指标到TensorBoard，支持在集群任何地方调用
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        value: 指标merge的值
        desc: 指标的描述，用于在TensorBoard中显示，字典的key可以为：
            - sum: 所有merge value的总和
            - cnt: 调用merge的次数
            - avg: 所有merge value的平均值，等于sum/cnt
            - time_window_sum: 最近一分钟的总和
            - time_window_cnt: 最近一分钟的次数
            - time_window_avg: 最近一分钟的平均值
    """
    metrics_worker = get_metrics_worker()
    metrics_worker.merge(name, Metric(1, value), desc, **kwargs)


def query(
    name: str, kind: str = "avg", time_window: bool = True, **kwargs
) -> float | int:
    """
    查询指标，支持在集群任何地方调用
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        kind: 指标查询的类型，可以为：
            - sum: 所有merge value的总和
            - cnt: 调用merge的次数
            - avg: 所有merge value的平均值，等于sum/cnt
        time_window: 是否启用时间窗口，如果启用，那么查询的是最近一分钟的指标
    """
    metrics_worker = get_metrics_worker()
    metric = metrics_worker.query(name, time_window, **kwargs)
    if kind == "avg":
        return float("nan") if metric.cnt == 0 else metric.avg
    elif kind == "sum":
        return metric.sum
    elif kind == "cnt":
        return metric.cnt
    else:
        raise ValueError(f"Unsupported kind: {kind}")
