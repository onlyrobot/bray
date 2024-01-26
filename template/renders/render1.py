import numpy as np
import matplotlib.pyplot as plt
import bray


def render(episode: list[bray.State], tick: int) -> bray.NestedArray:
    """
    渲染Episode中指定tick的画面，返回一个或多个表示图片的数组，
    图片的维度为(H,W,3)，数值范围为[0,255]
    Args:
        episode: Episode，是一个State的列表，State来自于AgentActor的tick
        tick_id: tick_id，表示当前要渲染的tick
    Returns:
        一个或多个np.ndarray，每个ndarray代表一张图片
    """
    image = episode[tick].transition["state"]["image"]
    return image.transpose(1, 2, 0)


def action_distribution(episode: list[bray.State], tick: int) -> bray.NestedArray:
    """
    渲染Episode中指定tick的动作分布，返回一个渲染后的动作分布图
    Args:
        episode: Episode，是一个State的列表，State来自于AgentActor的tick
        tick_id: tick_id，表示当前要渲染的tick
    Returns:
        一个np.ndarray，代表渲染后的动作分布图
    """
    actions = [s.transition["action"] for s in episode[: tick + 1]]
    action_space, action_names = 9, [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "UPRIGHT",
        "UPLEFT",
        "RIGHTFIRE",
        "LEFTIFIRE",
    ]
    hist, _ = np.histogram(actions, bins=range(action_space + 1))

    fig, ax = plt.subplots()
    ax.bar(action_names, hist / len(actions))
    # ax.set_xlabel('Actions')
    ax.set_ylabel("Probability")
    ax.set_title(f"Action Distribution before Tick {tick}")

    # 解决标签重叠问题
    ax.set_xticks(np.arange(action_space))
    ax.set_xticklabels(action_names, rotation=45, ha="right")

    fig.canvas.draw()  # 必须先绘制图表
    shape = fig.canvas.get_width_height()[::-1] + (3,)
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    plt.close(fig)  # 关闭图表释放资源
    return image.reshape(shape)