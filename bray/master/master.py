import asyncio
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import time
from datetime import datetime
import copy
from typing import Callable, Dict, Union

from concurrent.futures import ThreadPoolExecutor
import os, pickle


class Metric:
    def __init__(self, cnt=0, sum=0.0, step=None):
        self.cnt, self.sum, self.step = int(cnt), float(sum), step

    def merge(self, other: "Metric"):
        self.cnt += other.cnt
        self.sum += other.sum
        if self.step is None or other.step is None:
            return
        self.step = self.step + other.step

    def diff(self, other: "Metric") -> "Metric":
        if self.step is None or other.step is None:
            step = None
        else:
            step = self.step - other.step
        cnt, sum = self.cnt - other.cnt, self.sum - other.sum
        return Metric(cnt, sum, step)


@ray.remote(num_cpus=0, name="Master", get_if_exists=True)
class Master:
    def __init__(self, time_window: int):
        trial_path = ray.get_runtime_context().namespace

        self.registery, self.data, self.msgs = {}, {}, []
        self.log_path = os.path.join(trial_path, "bray.log")
        self.data_path = os.path.join(trial_path, "bray.pkl")
        if os.path.exists(self.data_path):
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        self.flush_cond = asyncio.Condition()

        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1))
        asyncio.create_task(self.flush())

        self.time_window = time_window
        self.metrics = self.get("metrics", {})
        self.last_metrics = copy.deepcopy(self.metrics)
        self.diff_metrics = {}
        self.step, self._get_global_step = self.get("step", -1), None
        self.descs = {}
        
        launch_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.launch_path = os.path.join(trial_path, launch_time)
        self.writer = None
        asyncio.create_task(self.start_tensorboard())

    def get_trial_launch_path(self) -> str: return self.launch_path

    def _init_writer(self):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            self.launch_path, flush_secs=self.time_window)

    async def start_tensorboard(self):
        await asyncio.sleep(self.time_window)
        if self.writer is None:
            self._init_writer()
        asyncio.create_task(self.dump_to_tensorboard())

    async def merge(self, name, metric, desc: Dict):
        if desc is not None:
            self.descs[name] = desc
        if m := self.metrics.get(name, None):
            m.merge(metric)
        else: self.metrics[name] = metric

    async def batch_merge(self, metrics: Dict):
        for name, metric in metrics.items():
            await self.merge(name, metric, None)

    def query(self, name, time_window: bool) -> Metric:
        if not time_window:
            return self.metrics.get(name, Metric())
        return self.diff_metrics.get(name, Metric())

    def batch_query(self, names, time_windows=True):
        if isinstance(time_windows, bool):
            time_windows = [time_windows] * len(names)
        return [self.query(n, t) for n, t in zip(names, time_windows)]

    async def _get_writer_and_step(self, step=None):
        if self.writer is None:
            self._init_writer()
        step = await self.get_global_step() if step is None else step
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
        writer.add_graph(model.requires_grad_(False).eval(), 
            input_to_model, use_strict_trace=False)

    def _dump_metric(self, name, metric):
        last = self.last_metrics.get(name, Metric())
        diff = metric.diff(last)
        step = self.step if diff.step is None else diff.step
        self.diff_metrics[name] = diff

        writer = self.writer
        if (desc := self.descs.get(name)) is None:
            if diff.cnt == 0: return
            avg = diff.sum / diff.cnt
            return writer.add_scalar(name, avg, step)
        if d := desc.get("sum"):
            writer.add_scalar(
                f"{name} -- {d}", metric.sum, step)
        if d := desc.get("cnt"):
            writer.add_scalar(
                f"{name} -- {d}", metric.cnt, step)
        if (d := desc.get("avg")) and metric.cnt:
            avg = metric.sum / metric.cnt
            writer.add_scalar(
                f"{name} -- {d}", avg, step)
        if d := desc.get("time_window_sum"):
            writer.add_scalar(
                f"{name} -- {d}", diff.sum, step
            )
        if d := desc.get("time_window_cnt"):
            writer.add_scalar(
                f"{name} -- {d}", diff.cnt, step
            )
        if (d := desc.get("time_window_avg")) and diff.cnt:
            avg = diff.sum / diff.cnt
            writer.add_scalar(f"{name} -- {d}", avg, step)

    async def dump_to_tensorboard(self):
        cur_metrics = copy.deepcopy(self.metrics)
        self.step = await self.get_global_step()
        self.set("step", self.step)

        for name, cur in cur_metrics.items():
            self._dump_metric(name, cur)
        self.last_metrics = cur_metrics
        self.set("metrics", cur_metrics)

        await asyncio.sleep(self.time_window)
        asyncio.create_task(self.dump_to_tensorboard())

    async def get_global_step(self) -> int:
        if not self._get_global_step:
            return self.step + 1
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, self._get_global_step)
        except Exception as e:
            print(f"Get step error: {e}")
        return self.step + 1

    async def set_tensorboard_step(self, get_global_step):
        self._get_global_step = get_global_step

    def set(self, key: str, value: object): 
        self.data[key] = value

    def get(self, key: str, default: object) -> object:
        return self.data.get(key, default)

    def register(self, key: str) -> int:
        self.registery[key] = self.registery.get(key, -1) + 1
        return self.registery[key]
    
    async def log(self, msg: str, flush: bool=False):
        timestamp = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime()
        )
        self.msgs.append(f"{timestamp} {msg}\n")
        if not flush: return
        async with self.flush_cond: self.flush_cond.notify()

    def _flush_log_and_data(self):
        self.msgs, msgs = [], self.msgs
        with open(self.data_path, "wb") as f:
            pickle.dump(self.data, f)
        with open(self.log_path, "+a") as f:
            f.writelines(msgs)
        
    async def wait_flush(self):
        try: await asyncio.wait_for(
            self.flush_cond.wait(), self.time_window)
        except asyncio.TimeoutError: pass
    
    async def flush(self):
        async with self.flush_cond: await self.wait_flush()
        loop = asyncio.get_running_loop()
        try: await loop.run_in_executor(
            None, self._flush_log_and_data,
        )
        except Exception as e:
            print(f"Log to {self.log_path} error: {e}")
        asyncio.create_task(self.flush())


def build_name(name, **kwargs) -> str:
    attributes = [f"{k}={v}" for k, v in kwargs.items()]
    return f"{name} {{{', '.join(attributes)}}}"


class Worker:
    global_cached_worker: "Worker" = None

    def __new__(cls, time_window: int = 60):
        if cls.global_cached_worker:
            return cls.global_cached_worker
        self = super().__new__(cls)
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False)

        self.master = Master.options(
            scheduling_strategy=scheduling_local,
        ).remote(time_window)

        self.cached_metrics = {}
        self.last_merge_time, self.merge_count = 0, 0
        self.merge_remote_interval = 1
        if not cls.global_cached_worker:
            cls.global_cached_worker = self
        return cls.global_cached_worker

    def __init__(self, time_window=60):
        """创建Worker作为Master的代理，调用全局Master的方法"""

    def flush_and_reset_merge_interval(self):
        merge_time = time.time()
        merge_time_interval = merge_time - self.last_merge_time

        merge_remote_interval = int((self.merge_count - 1) * 60 
            / merge_time_interval / 2)
        self.merge_remote_interval = 2 + min(
            self.merge_count + self.merge_count // 3,
            merge_remote_interval,
        )
        self.merge_count = 0
        self.last_merge_time = merge_time

        metrics, self.cached_metrics = self.cached_metrics, {}
        return self.master.batch_merge.remote(metrics)

    def merge(self, name, metric, desc, **kwargs):
        self.merge_count += 1
        if kwargs: name = build_name(name, **kwargs)
        m = self.cached_metrics.get(name, None)
        if not m:
            self.master.merge.remote(name, metric, desc)
            self.cached_metrics[name] = Metric()
        else: m.merge(metric)
        if self.merge_count < self.merge_remote_interval:
            return
        self.flush_and_reset_merge_interval()


def merge(
    name, value, step=None, desc: Dict = None, **kwargs
):
    """
    输出指标到TensorBoard，支持在集群任何地方调用
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        value: 指标merge的值
        step: 指标的横坐标，一个时间窗口内的step会取均值
        desc: 
    指标的描述，用于在TensorBoard中显示，字典的key可以为：
     - sum: 所有merge value的总和
     - cnt: 调用merge的次数
     - avg: 所有merge value的平均值，等于sum/cnt
     - time_window_sum: 最近一分钟的总和
     - time_window_cnt: 最近一分钟的次数
     - time_window_avg: 最近一分钟的平均值
    """
    Worker().merge(name, Metric(1, value, step), desc, **kwargs)


def add_scalar(name: str, value: float, step: int = None):
    """
    输出指标到TensorBoard，支持在集群任何地方调用，
    内部实现调用TensorBoard的add_scalar方法
    Args:
        name: 指标的名字
        value: 指标merge的值
        step: 指标的横坐标，默认使用全局的step
    """
    Worker().master.add_scalar.remote(name, float(value), step)


def add_image(
    name: str, image: object, step: int = None, dataformats="CHW"
):
    """
    输出图片到TensorBoard，支持在集群任何地方调用，
    内部实现调用TensorBoard的add_image方法
    Args:
        name: 图片的名字
        image: 图片的数据，可以为numpy数组或者torch tensor
        step: 图片的横坐标，默认使用全局的step
        dataformats: 图片的格式，默认为CHW
    """
    Worker().master.add_image.remote(name, image, step, dataformats)


def add_video(name: str, video: object, step: int = None, fps: int = 4):
    """
    输出视频到TensorBoard，支持在集群任何地方调用，
    内部实现调用TensorBoard的add_video方法
    Args:
        name: 视频的名字
        video: 视频的数据，可以为numpy数组或者torch tensor，
            维度为：(N,T,C,H,W)
        step: 视频的横坐标，默认使用全局的step
        fps: 视频的帧率，默认为4
    """
    Worker().master.add_video.remote(name, video, step, fps)


def add_histogram(
    name: str, values: object, step: int = None, bins: str = "tensorflow"
):
    """
    输出直方图到TensorBoard，支持在集群任何地方调用，
    内部实现调用TensorBoard的add_histogram方法
    Args:
        name: 直方图的名字
        values: 直方图的数据，可以为numpy数组或者torch tensor
        step: 直方图的横坐标，默认使用全局的step
        bins: 直方图的bins，默认为tensorflow
    """
    Worker().master.add_histogram.remote(name, values, step, bins)


def add_graph(model, input_to_model) -> ray.ObjectRef:
    """
    输出模型到TensorBoard，支持在集群任何地方调用，
    内部实现调用TensorBoard的add_graph方法
    Args:
        model: torch.nn.Module模型
        input_to_model: 模型的输入torch.Tensor或list of torch.Tensor
    """
    return Worker().master.add_graph.remote(model, input_to_model)


def get_global_step() -> int:
    """获取全局的step，支持在集群任何地方调用"""
    return ray.get(Worker().master.get_global_step.remote())


def query(
    name, kind: str = "avg", time_window=True, **kwargs
) -> Union[float, int]:
    """
    查询指标，支持在集群任何地方调用，由于是堵塞调用，所以不要太频繁
    Args:
        name: 指标的名字，配合 **kwargs 可以组成一个唯一的指标
        kind: 
    指标查询的类型，可以为：
     - sum: 所有merge value的总和
     - cnt: 调用merge的次数
     - avg: 所有merge value的平均值，等于sum/cnt
        time_window: 是否启用时间窗口，启用后查询的是最近一分钟的指标
    """
    if kwargs: name = build_name(name, **kwargs)
    metric = ray.get(
        Worker().master.query.remote(name, time_window)
    )
    if kind == "avg":
        if metric.cnt == 0: return float("nan")
        return metric.sum / metric.cnt
    return metric.sum if kind == "sum" else metric.cnt


def merge_time_ms(name, beg, step=None, **kwargs):
    value = (time.time() - beg) * 1000
    desc = {
        "time_window_avg": f"avg latency ms",
        "time_window_cnt": f"cnt per minute",
    }
    merge(name, value, step=step, desc=desc, **kwargs)


def set_tensorboard_step(
    remote_model=None, remote_buffer=None, get_global_step=None):
    """
    设置 TensorBoard 的全局 step，用于自定义 TensorBoard 的横坐标
    Args:
        remote_model: 绑定到指定RemoteModel的step
        get_global_step: 动态获取当前step的函数
    """
    if remote_model: get_global_step = lambda: remote_model.step
    if remote_buffer: 
        get_global_step = lambda: query(
            f"pop/{remote_buffer.name}", "sum", time_window=False)
    master = Worker().master
    ray.get(master.set_tensorboard_step.remote(get_global_step))


def get_trial_launch_path() -> str:
    """获取本次实验目录，目录结构： {trial_path}/{launch_time}"""
    return ray.get(Worker().master.get_trial_launch_path.remote())


def set(key: str, value: object):
    """将数据推送到全局的 Master 对象，比如config"""
    return Worker().master.set.remote(key, value)


def get(key: str, default: object=None) -> object:
    """从全局的 Master 对象获取数据"""
    return ray.get(Worker().master.get.remote(key, default))


def register(key: str) -> int:
    """注册一个全局的计数器，同步计数，比如 Actor 的 ID"""
    return ray.get(Worker().master.register.remote(key))

def log(msg: str, flush: bool=False):
    """输出日志到全局的 Master 对象，日志目录位于当前实验目录下"""
    return Worker().master.log.remote(msg, flush)


if __name__ == "__main__":
    ray.init(namespace="master", address="local")

    if not os.path.exists("master"):
        os.makedirs("master")

    worker = Worker(time_window=60)

    config = {"a": 1, "b": 2}
    ray.get(set("config", config))
    assert get("config") == config

    assert register("actor") == 0
    assert register("actor") == 1

    @ray.remote
    def test():
        assert get("config") == config
        assert register("actor") == 2
        ray.get(set("config", "hello"))

    config2 = {"a": 2, "b": 3}
    ray.get(test.remote())
    assert get("config") == "hello"
    assert register("actor") == 3
    ray.get(log("hello world 1", flush=True))
    ray.get(log("hello world 2", flush=True))
    ray.get(log("hello world 3", flush=True))
    ray.get(log("hello world 4", flush=True))

    import numpy as np
    print(get_trial_launch_path())
    for i in range(100):
        merge("test", i)
        add_scalar("test_scalar", i)
        add_image("test_image", np.random.rand(3, 32, 32), step=i)
        time.sleep(0.1)
    ray.get(worker.flush_and_reset_merge_interval())
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

    get_global_step = lambda: query("test1", kind="cnt", time_window=False)
    set_tensorboard_step(get_global_step=get_global_step)
    for i in range(100):
        merge("test1", i)
        merge("test2", i)
        add_scalar("test_scalar2", i)
        add_image("test_image2", np.random.rand(3, 32, 32))
        time.sleep(0.1)
    ray.get(worker.flush_and_reset_merge_interval())
    print("Test success!")
