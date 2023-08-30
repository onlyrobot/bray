from bray.buffer.buffer import RemoteBuffer
from bray.buffer.utils import (
    BatchBuffer,
    TorchTensorBuffer,
    PrefetchBuffer,
    CallbackBuffer,
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
from bray.utils.nested_array import NestedArray, handle_nested_array
from bray.metric.metric import (
    merge,
    query,
    get_metrics_worker,
    set_tensorboard_step,
)
from bray.metric.validate import Metric

import ray, logging

logger = logging.getLogger("ray")


def run_until_asked_to_stop():
    import signal

    signal.sigwait([signal.SIGTERM, signal.SIGINT])


def init(project: str, trial: str, **kwargs):
    """
    初始化 bray 运行环境，内部封装了 ray.init ，请在调用任何 bray 的API之前调用此函数
    Args:
        project: 项目名称，也就是项目的根目录
        trial: 试验名称，和 project 一起拼接为试验的根目录
        **kwargs: ray.init 的参数
    """
    import os

    trial_path = os.path.join(project, trial)
    trial_path = os.path.abspath(trial_path)
    if not os.path.exists(trial_path):
        os.makedirs(trial_path)

    ray.init(namespace=trial_path, dashboard_host="0.0.0.0", **kwargs)

    get_metrics_worker()  # 启动 RemoteMetrics 保证指标输出到Driver节点

    print("bray init success with path: ", trial_path)
