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
from bray.trainer.trainer import RemoteTrainer
from bray.actor.actor import RemoteActor
from bray.actor.base import Actor
from bray.agent.agent import State, AgentActor
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


CACHED_REMOTE_MODELS: dict[str:RemoteModel] = {}


async def forward(name: str, *args, **kwargs) -> NestedArray:
    """
    调用指定Model的 forward 方法，返回 forward 的结果
    Args:
        name: Model名称，如果不存在会抛出异常
        *args: 位置参数，类型为 NestedArray
        **kwargs: 关键字参数，类型为 NestedArray
    Returns:
        forward 的结果，类型为 NestedArray
    """
    global CACHED_REMOTE_MODELS
    if name not in CACHED_REMOTE_MODELS:
        CACHED_REMOTE_MODELS[name] = RemoteModel(name)
    remote_model = CACHED_REMOTE_MODELS[name]
    return await remote_model.forward(*args, **kwargs)


Buffer = RemoteBuffer
CACHED_REMOTE_BUFFERS: dict[str:RemoteBuffer] = {}


def push(name, *args: NestedArray):
    """
    调用指定Buffer的 push 方法，将数据推送到缓冲区
    Args:
        name: Buffer名称，如果不存在会抛出异常
        *args: 位置参数，类型为 NestedArray
    """
    global CACHED_REMOTE_BUFFERS
    if name not in CACHED_REMOTE_BUFFERS:
        CACHED_REMOTE_BUFFERS[name] = RemoteBuffer(name)
    remote_buffer = CACHED_REMOTE_BUFFERS[name]
    return remote_buffer.push(*args)


def run_until_asked_to_stop():
    import signal

    signal.sigwait([signal.SIGTERM, signal.SIGINT])
