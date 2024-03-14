import asyncio
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import time
from datetime import datetime
import copy
from typing import Callable
from bray.master.master import Master

from concurrent.futures import ThreadPoolExecutor


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


@ray.remote(num_cpus=0)
class Metrics(Master):
    def __init__(self, time_window):
        super().__init__(time_window)
        self.metrics, self.last_metrics = {}, {}
        self.diff_metrics = {}
        self.descs = {}
        self.step, self.time_window = -1, time_window
        self.step_model, self._get_step = None, None

        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )

        trial_path = ray.get_runtime_context().namespace
        launch_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.launch_path = f"{trial_path}/{launch_time}"
        self.writer = None

        asyncio.create_task(self.start_tensorboard())

    def _init_writer(self):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            self.launch_path,
            flush_secs=self.time_window,
        )

    def get_trial_launch_path(self) -> str:
        return self.launch_path

    async def start_tensorboard(self):
        await asyncio.sleep(self.time_window)
        if self.writer is None:
            self._init_writer()
        asyncio.create_task(self.dump_to_tensorboard())

    async def merge(self, name, metric, desc: dict[str:str]):
        if desc is not None:
            self.descs[name] = desc
        m = self.metrics.get(name, None)
        if m:
            m.merge(metric)
        else:
            self.metrics[name] = metric

    async def batch_merge(self, metrics: dict[str:Metric]):
        for name, metric in metrics.items():
            await self.merge(name, metric, None)

    async def _get_writer_and_step(self, step=None):
        if self.writer is None:
            self._init_writer()
        step = await self.get_step() if step is None else step
        return self.writer, step

    async def add_scalar(self, name, value, step):
        writer, step = await self._get_writer_and_step(step)
        writer.add_scalar(name, value, step)

    async def add_image(self, name, image, step, dataformats):
        writer, step = await self._get_writer_and_step(step)
        writer.add_image(name, image, step, dataformats=dataformats)

    async def add_video(self, name, video, step, fps):
        writer, step = await self._get_writer_and_step(step)
        writer.add_video(name, video, step, fps)

    async def add_histogram(self, name, values, step, bins):
        writer, step = await self._get_writer_and_step(step)
        writer.add_histogram(name, values, step, bins)

    async def add_graph(self, model, input_to_model):
        writer, _ = await self._get_writer_and_step()
        writer.add_graph(model.eval(), input_to_model, use_strict_trace=False)

    async def query(self, name, time_window: bool) -> Metric:
        if not time_window:
            return self.metrics.get(name, Metric())
        return self.diff_metrics.get(name, Metric())

    def _dump_by_desc(self, name, metric, diff):
        desc = self.descs.get(name, None)

        if desc is None:
            if diff.cnt != 0:
                self.writer.add_scalar(name, diff.avg, self.step)
            return
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
        current_metrics = copy.deepcopy(self.metrics)
        self.step = await self.get_step()

        for name, current in current_metrics.items():
            last = self.last_metrics.get(name, Metric())
            diff = Metric(
                current.cnt - last.cnt,
                current.sum - last.sum,
            )
            # diff metrics used for time window query
            self.diff_metrics[name] = diff

            self._dump_by_desc(name, current, diff)

        self.last_metrics = current_metrics

        await asyncio.sleep(self.time_window)
        asyncio.create_task(self.dump_to_tensorboard())

    async def get_step(self) -> int:
        if self.step_model:
            return await self.step_model.get_step.remote(self.model_name)
        if not self._get_step:
            return self.step + 1
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                self._get_step,
            )
        except Exception as e:
            print(f"Get step error: {e}")
            return self.step + 1

    async def set_tensorboard_step(self, model: str, get_step: Callable):
        if model:
            self.step_model = ray.get_actor(model.split("/")[0])
            self.model_name = model
        self._get_step = get_step


def build_name(name, **kwargs) -> str:
    attributes = [f"{k}={v}" for k, v in kwargs.items()]
    return f"{name} {{{', '.join(attributes)}}}"


class MetricsWorker:
    def __init__(self, time_window):
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self.remote_metrics = Metrics.options(
            name="Metrics",
            get_if_exists=True,
            scheduling_strategy=scheduling_local,
        ).remote(time_window)

        self.cached_metrics = {}
        self.last_merge_time = 0
        self.merge_remote_interval, self.merge_count = 1, 0

    def flush_and_reset_merge_interval(self):
        merge_time = time.time()
        merge_time_interval = merge_time - self.last_merge_time

        self.merge_remote_interval = 2 + min(
            self.merge_count + self.merge_count // 2,
            int((self.merge_count - 1) * 60 / merge_time_interval),
        )
        self.merge_count = 0
        self.last_merge_time = merge_time

        metrics, self.cached_metrics = self.cached_metrics, {}
        return self.remote_metrics.batch_merge.remote(metrics)

    def merge(self, name, metric, desc, **kwargs):
        self.merge_count += 1
        if kwargs:
            name = build_name(name, **kwargs)
        m = self.cached_metrics.get(name, None)
        if not m:
            self.remote_metrics.merge.remote(name, metric, desc)
            self.cached_metrics[name] = Metric()
        else:
            m.merge(metric)
        if self.merge_count < self.merge_remote_interval:
            return
        self.flush_and_reset_merge_interval()


METRICS_WORKER = None


def get_metrics_worker(time_window=60) -> MetricsWorker:
    global METRICS_WORKER
    if METRICS_WORKER is None:
        METRICS_WORKER = MetricsWorker(time_window)
    return METRICS_WORKER


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
    get_metrics_worker().merge(name, Metric(1, value), desc, **kwargs)


def add_scalar(name: str, value: float, step: int = None):
    """
    输出指标到TensorBoard，支持在集群任何地方调用，
    该方法就是直接调用TensorBoard的add_scalar方法
    Args:
        name: 指标的名字
        value: 指标merge的值
        step: 指标的横坐标，默认使用全局的step
    """
    get_metrics_worker().remote_metrics.add_scalar.remote(name, value, step)


def add_image(name: str, image: object, step: int = None, dataformats="CHW"):
    """
    输出图片到TensorBoard，支持在集群任何地方调用，
    该方法就是直接调用TensorBoard的add_image方法
    Args:
        name: 图片的名字
        image: 图片的数据，可以为numpy数组或者torch tensor
        step: 图片的横坐标，默认使用全局的step
        dataformats: 图片的格式，默认为CHW
    """
    remote_metrics = get_metrics_worker().remote_metrics
    remote_metrics.add_image.remote(name, image, step, dataformats)


def add_video(name: str, video: object, step: int = None, fps: int = 4):
    """
    输出视频到TensorBoard，支持在集群任何地方调用，
    该方法就是直接调用TensorBoard的add_video方法
    Args:
        name: 视频的名字
        video: 视频的数据，可以为numpy数组或者torch tensor，
            维度为：(N,T,C,H,W)
        step: 视频的横坐标，默认使用全局的step
        fps: 视频的帧率，默认为4
    """
    remote_metrics = get_metrics_worker().remote_metrics
    remote_metrics.add_video.remote(name, video, step, fps)


def add_histogram(
    name: str, values: object, step: int = None, bins: str = "tensorflow"
):
    """
    输出直方图到TensorBoard，支持在集群任何地方调用，
    该方法就是直接调用TensorBoard的add_histogram方法
    Args:
        name: 直方图的名字
        values: 直方图的数据，可以为numpy数组或者torch tensor
        step: 直方图的横坐标，默认使用全局的step
        bins: 直方图的bins，默认为tensorflow
    """
    remote_metrics = get_metrics_worker().remote_metrics
    remote_metrics.add_histogram.remote(name, values, step, bins)


def add_graph(model, input_to_model):
    """
    输出模型到TensorBoard，支持在集群任何地方调用，
    该方法就是直接调用TensorBoard的add_graph方法
    Args:
        model: torch.nn.Module模型
        input_to_model: 模型的输入torch.Tensor或list of torch.Tensor
    """
    remote_metrics = get_metrics_worker().remote_metrics
    remote_metrics.add_graph.remote(model, input_to_model)


def get_step() -> int:
    """获取全局的step，支持在集群任何地方调用"""
    remote_metrics = get_metrics_worker().remote_metrics
    return ray.get(remote_metrics.get_step.remote())


def query(
    name: str, kind: str = "avg", time_window: bool = True, **kwargs
) -> float | int:
    """
    查询指标，支持在集群任何地方调用，由于是堵塞调用，所以不要太频繁
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        kind: 指标查询的类型，可以为：
            - sum: 所有merge value的总和
            - cnt: 调用merge的次数
            - avg: 所有merge value的平均值，等于sum/cnt
        time_window: 是否启用时间窗口，如果启用，那么查询的是最近一分钟的指标
    """
    remote_metrics = get_metrics_worker().remote_metrics
    if kwargs:
        name = build_name(name, **kwargs)
    metric = ray.get(remote_metrics.query.remote(name, time_window))
    if kind == "avg":
        return metric.avg if metric.cnt else float("nan")
    if kind == "sum":
        return metric.sum
    if kind == "cnt":
        return metric.cnt
    raise ValueError(f"Unsupported kind: {kind}")


def merge_time_ms(name, beg, **kwargs):
    merge(
        name,
        (time.time() - beg) * 1000,
        desc={
            "time_window_avg": f"avg latency ms",
            "time_window_cnt": f"cnt per minute",
        },
        **kwargs,
    )


def set_tensorboard_step(model: str = None, get_step: Callable = None):
    """
    设置 TensorBoard 的 step，用于自定义 TensorBoard 的横坐标
    Args:
        model: RemoteModel的名称，Tensorboard的横坐标将被设置为该模型的step
        get_step: 一个函数，用于获取当前的TensorBoard的横坐标，比如：

            将TensorBoard的横坐标设置为tick数：
        ` get_step = lambda: bray.query("tick", kind="cnt", time_window=False) `

            将Tensorboard的横坐标设为指定buffer的pop数：
        ` get_step = lambda: bray.query(
            "pop", kind="sum", time_window=False, buffer="my_buffer") `
    """
    remote_metrics = get_metrics_worker().remote_metrics
    ray.get(remote_metrics.set_tensorboard_step.remote(model, get_step))


def get_trial_launch_path() -> str:
    """
    获取本次实验的TensorBoard的目录，目录结构： {trial_path}/{launch_time}
    """
    remote_metrics = get_metrics_worker().remote_metrics
    return ray.get(remote_metrics.get_trial_launch_path.remote())


if __name__ == "__main__":
    import time
    import numpy as np
    import ray

    ray.init(namespace="test/metrics")
    get_metrics_worker(time_window=1)
    print(get_trial_launch_path())
    for i in range(100):
        merge("test", i)
        add_scalar("test_scalar", i)
        add_image("test_image", np.random.rand(3, 32, 32), step=i)
        time.sleep(0.1)
    ray.get(get_metrics_worker().flush_and_reset_merge_interval())
    assert query("test", kind="sum", time_window=False) == 4950
    assert query("test", kind="avg", time_window=False) == 49.5
    assert query("test", kind="cnt", time_window=False) == 100

    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    add_graph(model, torch.rand(1, 1))

    get_step = lambda: query("test1", kind="cnt", time_window=False)
    set_tensorboard_step(get_step=get_step)
    for i in range(100):
        merge("test1", i)
        merge("test2", i)
        add_scalar("test_scalar2", i)
        add_image("test_image2", np.random.rand(3, 32, 32))
        time.sleep(0.1)
    ray.get(get_metrics_worker().flush_and_reset_merge_interval())
    print("Test success!")