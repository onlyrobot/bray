# Bray框架模板介绍
除了使用Bray提供的基本分布式组件（RemoteModel、RemoteActor等）搭建机器学习流程以外，还可以直接基于预定义的模板快速进行模仿学习、强化学习、模仿+强化学习以及服务部署的开发。
## 一、模板的定义
模板是高于组件的抽象，提供了参数化的启动方式，能够加快项目接入流程，减少出错的可能性，此外多个项目用同一个模板，也可以很好地复用代码，提高可维护性。
Bray的所有模板都是基于以下6个组件构建的，组件之间充分地解耦，可以独立地开发和验证，一个模板可以包含任意类型和数量的组件：
1. Model：模型，指的是PyTorch模型，封装为RemoteModel后支持高效推理和检查点管理
2. Agent：代表了游戏中具备一定决策和动作能力的智能体，多个智能体之间可以灵活的交互
3. Buffer：缓冲区，也是强化学习中的经验回放池，支持不同采样算法，支持添加不同数据源
4. Source：数据源，可以是文件、网络、对象存储等的封装，添加到缓存区后用于采样训练
5. Actor：封装了智能体和环境的交互逻辑，也是部署时对外暴露的AI服务接口
6. Trainer：封装了模型、缓存区和训练算法，支持分布式拓展
目前Bray中定义了三种机器学习的模板：强化学习、模仿学习、模仿+强化学习，分别对应到[这个代码目录](../template)下的三个配置文件。
## 二、模板示例
以模仿+强化学习模板为例，它的[配置参数](../template/config.yaml)为：
```yaml
project: template  # 项目名称，此处为模板
trial: train_sl_rl_v0  # 实验名称，项目和实验共同确定一个命名空间
mode: train  # 启动模式，此处为训练模式
dump_state: true  # 是否保存轨迹的状态，用于渲染和回放

# 模型配置 - 定义了一个PyTorch模型的具体参数和设置
model1:  
  kind: model  # 组件类型：模型
  module: template.models.model1  # 模型的模块路径，确保该模块下包含 build_model 函数
  checkpoint_interval: null  # 模型检查点保存间隔，单位step，null表示10分钟保存一次
  checkpoint: null  # 加载的检查点路径或者step编号，null表示加载最新的检查点
  max_batch_size: 1  # 模型推理的最大批次大小
  num_workers: 0  # 工作进程数
  cpus_per_worker: 0.5  # 每个工作进程分配的CPU数量
  gpus_per_worker: 0.0  # 每个工作进程分配的GPU数量
  memory_per_worker: 1024  # 每个工作进程分配的内存量（以MB为单位）
  use_onnx: train  # 是否使用ONNX格式，可选值为[train, infer, quantize, null]
  local_mode: true  # 是否在本地模式下运行，开启后模型将部署在调用方本地
  override_model: true  # 多次实验时，是否覆盖已存在的模型，改为false可以加速启动
  tensorboard_graph: true  # 是否将模型可视化到TensorBoard中
  tensorboard_step: true  # 是否将TensorBoard的横坐标设为模型的step

# 评估数据源配置 - 定义了评估数据的来源和设置
eval_source:
  kind: source  # 组件类型：数据源
  enable: train  # 启用状态，此处为训练模式下启用
  module: template.sources.source1  # 评估数据源的模块路径
  func: build_eval_source  # 模块路径下构建数据源的函数
  num_workers: null  # 工作进程数，null表示默认值
  epoch: 10000  # 迭代周期数

# 训练数据源配置 - 定义了训练数据的来源和设置
train_source:
  kind: source  # 组件类型：数据源
  enable: train  # 启用状态，此处为训练模式下启用
  module: template.sources.source1  # 训练数据源的模块路径
  func: build_source  # 模块路径下构建数据源的函数
  num_workers: null  # 工作进程数，null表示默认值
  epoch: 100  # 迭代周期数

# 缓存区配置 - 定义了评估数据缓存的设置
eval_buffer:
  kind: buffer  # 组件类型：缓冲区
  enable: train  # 启用状态，此处为训练模式下启用
  sources:
    - eval_source  # 添加的数据源名称
  size: 8  # 缓冲区大小，单位为批次数量，过大容易导致OOM
  batch_size: 128  # 批量大小，null表示不组批次
  num_workers: 1  # 工作进程数，一般为1即可
  density: 100  # 数据生产和消费端连接密度，调小可以减少系统资源占用

# 缓存区配置 - 定义了训练数据缓存的设置
train_buffer:
  kind: buffer  # 组件类型：缓冲区
  enable: train  # 启用状态，此处为训练模式下启用
  sources:
    - train_source  # 添加的数据源名称
  size: 8  # 缓冲区大小，单位为批次数量，过大容易导致OOM
  batch_size: 128  # 批量大小，null表示不组批次
  num_workers: 1  # 工作进程数，一般为1即可
  density: 100  # 数据生产和消费端连接密度，调小可以减少系统资源占用

# 缓存区配置 - 定义了强化学习经验池的设置
buffer1:
  kind: buffer  # 组件类型：缓冲区
  enable: train  # 启用状态，此处为训练模式下启用
  size: 8  # 缓冲区大小，单位为批次数量，过大容易导致OOM
  batch_size: 128  # 批量大小，null表示不组批次
  num_workers: 2  # 工作进程数，一般为1~2
  density: 100  # 数据生产和消费端连接密度，调小可以减少系统资源占用

# 智能体配置 - 定义了负责决策和动作的主智能体
agent1:
  kind: agent  # 组件类型：智能体
  module: template.agents.agent1  # 智能体的模块路径
  class: Agent1  # 智能体类的名称

# 智能体配置 - 定义了负责实时统计输出指标的智能体 
metrics_agent:
  kind: agent  # 组件类型：智能体
  module: template.agents.metrics_agent  # 智能体的模块路径
  class: MetricsAgent  # 智能体类的名称

# Actor配置 - 一个Actor封装了了智能体和环境交互逻辑
actor1:
  kind: actor  # 组件类型：Actor
  port: 8000  # 环境通过该端口和Actor交互，也就是AI服务的端口号
  num_workers: 2  # 工作进程数
  actors_per_worker: 10  # 每个工作进程的Actor数量，用于控制并发
  cpus_per_worker: 1.0  # 每个工作进程分配的CPU数量
  memory_per_worker: 512  # 每个工作进程分配的内存量（以MB为单位）
  use_tcp: false  # 是否使用TCP协议，默认为HTTP协议
  use_gateway: node  # 使用的网关类型，可选值为[node, head, null]
  agents:  # 包含的智能体
    - agent1
    - metrics_agent
  episode_length: 128  # 单次轨迹的长度，null表示不限制，取决于游戏何时结束
  episode_save_interval: 1000 # 轨迹的保存间隔，单位次，null表示不保存
  serialize: json  # 序列化器，null表示不使用，可选值为["proto", "json"]
  tick_input_proto:  # Actor的tick输入protobuf数据定义
    module: template.tick_pb2  # 编译好的protobuf文件模块路径
    message: TickInput  # 在模块路径下具体的Message类
  tick_output_proto:  # Actor的tick输出protobuf数据定义
    module: template.tick_pb2  # 编译好的protobuf文件模块路径
    message: TickOutput  # 在模块路径下具体的Message类

# 渲染器配置 - 渲染器负责将轨迹渲染为图片或视频
render1:
  kind: render  # 组件类型：渲染器
  module: template.renders.render1  # 渲染器的模块路径
  func: render  # 模块路径下渲染器的函数，默认为 render

# 渲染器配置 - 渲染器负责将轨迹渲染为图片或视频
action_render:
  kind: render  # 组件类型：渲染器
  module: template.renders.render1  # 渲染器的模块路径
  func: action_distribution  # 模块路径下渲染器的函数，默认为 render

# 训练器配置 - 定义了一个训练器的设置
trainer1:
  kind: trainer  # 组件类型：训练器
  enable: train  # 启用状态，此处为训练时启用
  module: template.trainers.sl_rl_trainer  # 训练器的模块路径
  class: Trainer1  # 训练器的类名称
  model: model1  # 被训练的模型
  buffers:  # 训练用的缓存区列表
    - train_buffer
    - buffer1
  buffer_weights:  # 每个训练用缓存区对应的采样权重
    - 0.5
    - 0.5
  eval:  # 评估相关的配置信息
    buffer: eval_buffer  # 评估使用的缓冲区
    interval: 1000  # 评估间隔，单位step
    steps: 10  # 每次评估的步数
  use_gpu: null  # 是否使用GPU，null表示不使用
  num_workers: null  # 工作进程数，null表示默认值
  cpus_per_worker: null  # 每个工作进程分配的CPU数量，null表示默认设置
  batch_size: 1  # 训练用的批次大小，这是在缓存区的批次大小基础上再次组批次
  batch_kind: concate  # 组批次的方式，可选值为["concate", "stack", null]
  prefetch_size: 1  # 预取批次大小，为0表示不预取，一般设为1
  max_reuse: 1  # 样本最大重用次数，和预取批次大小结合使用
  learning_rate: null  # 优化器的学习率，默认值null代表5e-4
  clip_grad_max_norm: 1.0  # 梯度裁剪的最大范数
  weights_publish_interval: 1  # 权重发布间隔，强化学习的实时性较高
  num_steps: 10000000  # 总训练步数

# 模型网络配置 - 定义了模型的输入和输出空间
network:
  kind: config  # 组件类型：配置
  state_space: [4, 42, 42]  # 输入空间：[通道数, 高度, 宽度]
  action_space: 9 # 输出空间：[动作数量]
```
启动模仿+强化训练的命令：
```bash
python -m bray.launch --config template/sl_rl.yaml
```
可以通过--help选项查看launch的参数：
```bash
python -m bray.launch --help
```
模板中每个组件的详细定义可以[对应目录](../template/)下的代码，以模仿训练的[Source](../template/sources/source1.py)为例，定义如下：
```python
from typing import Iterator, Iterable
import bray
import numpy as np


class AtariDataset(Iterable[bray.NestedArray]):
    def __init__(self, size=10000):
        self.fake_data = {
            "state": {"image": np.random.randn(4, 42, 42).astype(np.float32)},
            "action": np.random.randint(0, 9),
        }
        self.size = size

    def __iter__(self) -> Iterator[bray.NestedArray]:
        """
        返回一个迭代器，本方法可能会被多次调用，每次调用都会返回一个新的迭代器。
        """
        return self.generate()

    def generate(self) -> Iterator[bray.NestedArray]:
        for _ in range(self.size):
            yield self.fake_data


def build_source(name, config: dict) -> list[Iterable[bray.NestedArray]]:
    """
    构建数据源，返回一个列表，列表中的每个元素都是一个可迭代数据集，
    返回数据集数量越多，生成数据的并行度越高
    Args:
        name: 数据源名称，在配置文件中指定
        config: 全局配置，通过 `config[name]` 获取当前数据源配置
    Returns:
        sources: 数据源列表，每个元素都是独立的数据集
    """
    return [AtariDataset() for _ in range(2)]


def build_eval_source(name, config: dict) -> list[Iterator[bray.NestedArray]]:
    return [AtariDataset() for _ in range(1)]
```

## 三、如何使用模板
新增项目时，将[template](../template/)复制为项目名称，修改其中的配置文件和对应组件的定义，即可开始训练。注意当前template目录包含了模仿和强化的所有组件，可以根据具体情况进行删除和修改，比如：
模仿学习只需要定制化组件Model、Source、Trainer，在部署阶段再用到Actor；强化学习需要定制化组件Model、Agent、Trainer。
具体的组件实现可以参考模板中的示例，请关注它的注释和继承的基类。已有的不是基于模板实现的项目，可以考虑迁移过来，方便使用到Bray的最新功能和特性。
## 四、未来规划
得益于模板的抽象，可以通过组件化的方式进行机器学习，后续会新增更多的机器学习模板，比如League训练的模板。
对于一些常用的，简单的模板，后续会加入可视化建模的功能，提供更加直观的方式来搭建训练流程，并集成Tensorboard、Notebook等可视化工具。