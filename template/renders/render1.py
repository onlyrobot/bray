import bray

def render(episode: list[bray.State], tick_id: int) -> bray.NestedArray:
    """
    渲染Episode中指定tick的画面，返回一个或多个表示图片的数组，
    图片的维度为(H,W,3)，数值范围为[0,255]
    Args:
        episode: Episode，是一个State的列表，State来自于AgentActor的tick
        tick_id: tick_id，表示当前要渲染的tick
    Returns:
        一个或多个np.ndarray，每个ndarray代表一张图片
    """
    image = episode[tick_id].transition["state"]["image"]
    return image.transpose(1, 2, 0)