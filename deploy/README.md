# 模型导出、校验、部署框架

本框架是Bray的子模块，用于将基于Bray训练的模型无缝导出到不同推理框架框架（比如ONNX、MNN），并在不同设备（比如Windows、Android）和编程语言（比如C#、Python）上进行自动正确性校验和性能分析。

模型训练到部署的迁移过程往往会遇到算子不兼容、量化误差过高等问题，为了提前暴露这类潜在风险，在定义模型网络结构阶段就应该执行一遍模型导出、校验、部署的Pipeline。

## 使用方法

1. 定义PyTorch模型，并接入到Bray的RemoteModel，接入方式参考 [Bray介绍](../README.md)

```python
bray.init(project="./atari", trial="deploy")
"""
以下代码成功执行后，会在./atari目录下生成以下目录结构：
deploy
├── atari_model
│   ├── atari_model.pt  # 原始的PyTorch模型
│   └── model.onnx  # 自动转换的onnx模型
|   └── forward_inputs.pt   # 模型输入参数，也就是 [forward_args, forward_kwargs]
|   └── forward_outputs.pt  # 模型的原始输出
|   └── weights.pt  # 模型权重
└── xxx
"""
remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=(np.random.randn(42, 42, 4).astype(np.float32),),
    use_onnx="infer",
)
```

2. 基于Bray的RemoteModel生成文件路径，执行Pipeline流程

```bash
# 成功执行后会输出模型在不同推理框架上的推理延迟、相对误差等等
python -m deploy.launch ./atari/deploy/atari_model
```

## 支持功能列表

1. Onnx框架

|    | Linux | Window | Android |
| --- | --- | --- | --- |
| Python | 完成 | 完成 | / |
| C# | 完成 | 完成 | / |

1. MNN框架

|     | Linux | Window | Android |
| --- | --- | --- | --- |
| Python | / | / | / |
| C# | / | / | / |