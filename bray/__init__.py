from bray.buffer.buffer import RemoteBuffer
from bray.buffer.utils import (
    BatchBuffer,
    TorchTensorBuffer,
    ReuseBuffer,
    PrefetchBuffer,
    CallbackBuffer,
    TensorFlowTensorBuffer,
    SampleBuffer,
)
from bray.model.model import (
    RemoteModel,
    get_torch_model_weights,
    set_torch_model_weights,
)
from bray.model.onnx import export_onnx
from bray.trainer.trainer import RemoteTrainer, train
from bray.trainer.base import Trainer
from bray.actor.actor import RemoteActor
from bray.actor.base import Actor
from bray.actor.agent import State, Agent, AgentActor
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    make_batch,
)
from bray.metric.metric import (
    merge,
    query,
    add_scalar,
    add_image,
    add_video,
    add_histogram,
    add_graph,
    get_metrics_worker,
    set_tensorboard_step,
    get_trial_launch_path,
)
from bray.metric.validate import Metric
from bray.master.master import (
    set,
    get,
    register,
)

import ray, logging, os

logger = logging.getLogger("ray")


def init(project: str, trial: str, **kwargs):
    """
    初始化 bray 运行环境，内部封装了 ray.init ，请在调用任何
      bray 的API之前调用此函数
    Args:
        project: 项目名称，也就是项目的根目录
        trial: 试验名称，和 project 一起拼接为试验的根目录
        **kwargs: ray.init 的关键字参数
    """
    trial_path = os.path.join(project, trial)
    trial_path = os.path.abspath(trial_path)

    if not os.path.exists(trial_path):
        os.makedirs(trial_path)

    ray.init(namespace=trial_path, dashboard_host="0.0.0.0", **kwargs)

    # 启动 Metrics 保证指标输出到Driver节点
    get_metrics_worker()

    print("bray init success with path: ", trial_path)


def run_until_asked_to_stop():
    import signal

    signal.sigwait([signal.SIGTERM, signal.SIGINT])


async def forward(name: str, *args, batch=False, **kwargs) -> NestedArray:
    """
    调用指定Model的 forward 方法，返回 forward 结果，注意batch维度的处理
    Args:
        name: Model名称，如果不存在会抛出异常
        *args: 位置参数，类型为 NestedArray
        batch: 输入和输出是否包含batch维度，默认不包含
        **kwargs: 关键字参数，类型为 NestedArray
    Returns:
        forward 的结果，类型为 NestedArray
    """
    return await RemoteModel(name).forward(*args, batch=batch, **kwargs)


def push(name: str, *args: NestedArray, drop=True):
    """
    调用指定Buffer的 push 方法，将一个或多个数据推送到缓冲区
    Args:
        name: Buffer名称，如果不存在会抛出异常
        *args: 位置参数，类型为 NestedArray，其中的每个元素都会被推送到缓冲区
        drop: 是否丢弃数据，如果为True，当缓冲区满时，会丢弃最早的数据，
            否则当缓冲区满时，会阻塞直到缓冲区有空间
    """
    return RemoteBuffer(name).push(*args)


def pop(name: str) -> NestedArray:
    """
    从指定Buffer中弹出一个数据，如果缓冲区为空，会阻塞直到缓冲区有数据
    Args:
        name: Buffer名称，如果不存在会抛出异常
    Returns:
        从缓冲区中弹出的数据，类型为 NestedArray
    """
    return next(RemoteBuffer(name))
