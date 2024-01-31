import gradio as gr
import os, yaml, argparse, importlib
import bray
import pickle
import time

parser = argparse.ArgumentParser(description="Launch Bray render")
parser.add_argument("--config", help="Config yaml file", required=True)
args = parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


CONFIG = load_config(args.config)


def build_trials() -> list[str]:
    def is_trial(trial: str):
        if not os.path.isdir(path := os.path.join(CONFIG["project"], trial)):
            return False
        return "episode" in os.listdir(path)

    return [t for t in os.listdir(CONFIG["project"]) if is_trial(t)]


def build_episodes(trial: str) -> list[str]:
    episodes = os.listdir(os.path.join(CONFIG["project"], trial, "episode"))
    return sorted(episodes, key=lambda x: int(x.split("-")[1]), reverse=True)


def build_renders() -> dict[str, callable]:
    is_render = lambda c: isinstance(c, dict) and c.get("kind") == "render"
    return [n for n in CONFIG if is_render(CONFIG[n])] + ["raw"]


def select_trial(trial: str):
    return gr.update(choices=build_episodes(trial))


def select_episode(trial: str, episode: str, tick: int):
    episode_path = os.path.join(CONFIG["project"], trial, "episode", episode)
    max_tick = len(os.listdir(episode_path))
    tick = tick if tick < max_tick else 0
    return gr.update(maximum=max_tick - 1, value=tick), max_tick


def build_image(trial: str, episode: str, tick: int, render: str):
    hidden, visible = gr.update(visible=False), gr.update(visible=True)
    if not trial or not episode or not render:
        return visible, hidden

    def load_state(path: str, tick: int) -> bray.State:
        path = os.path.join(path, f"tick-{tick}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def build_episode(path: str) -> list[bray.State]:
        ticks = [n.split(".")[0].split("-")[1] for n in os.listdir(path)]
        ticks.sort(key=lambda x: int(x))
        return [load_state(path, int(t)) for t in ticks]

    path = os.path.join(CONFIG["project"], trial, "episode", episode)
    eps = build_episode(path)
    if render == "raw":
        return hidden, gr.update(value=str(eps[tick]), visible=True)
    module = importlib.import_module(CONFIG[render]["module"])
    module = importlib.reload(module)
    func = getattr(module, CONFIG[render].get("func", "render"))
    return gr.update(value=func(eps, tick), visible=True), hidden


with gr.Blocks() as app:
    title = gr.Label(
        "Bray Render For " + CONFIG["project"].capitalize(),
    )
    with gr.Row():
        trial = gr.Dropdown(label="Select Trial")
        episode = gr.Dropdown(
            label="Select Episode",
        )
        render = gr.Dropdown(label="Select Render")
        speed = gr.Slider(
            minimum=0,
            maximum=10,
            step=0.1,
            label="Play Speed",
        )

    def dynamic_reload_config():
        global CONFIG
        CONFIG = load_config(args.config)
        trial = gr.update(
            choices=build_trials(),
            value=CONFIG["trial"],
        )
        episode = gr.update(
            choices=build_episodes(CONFIG["trial"]),
        )
        render = gr.update(
            choices=build_renders(),
        )
        return trial, episode, render

    app.load(
        dynamic_reload_config,
        inputs=None,
        outputs=[trial, episode, render],
    )
    tick = gr.Slider(label="Select Tick")
    max_tick = gr.Number(visible=False)
    image = gr.Image(label="Render Image")
    markdown = gr.Markdown(visible=False)
    trial.change(select_trial, inputs=[trial], outputs=[episode])
    episode.change(
        select_episode,
        inputs=[trial, episode, tick],
        outputs=[tick, max_tick],
        show_progress="hidden",
    )
    render.change(
        build_image,
        inputs=[trial, episode, tick, render],
        outputs=[image, markdown],
        show_progress="hidden",
    )
    tick.change(
        build_image,
        inputs=[trial, episode, tick, render],
        outputs=[image, markdown],
        show_progress="hidden",
    )

    def count(speed, player):
        time.sleep(1 / speed if speed else 0.2)
        return time.time(), player + speed

    player = gr.Number(visible=False)
    player.change(
        lambda t, max_t: min(t + 1, max_t - 1),
        inputs=[tick, max_tick],
        outputs=[tick],
        show_progress="hidden",
    )
    counter = gr.Number(visible=False)
    counter.change(
        count,
        inputs=[speed, player],
        outputs=[counter, player],
        show_progress="hidden",
    )
    app.load(
        lambda: time.time(),
        inputs=None,
        outputs=[counter],
        show_progress="hidden",
    )
app.queue().launch(server_name="0.0.0.0", server_port=7860)
