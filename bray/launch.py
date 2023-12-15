import yaml, importlib, argparse
import bray

MODELS, SOURCES, BUFFERS, TRAINERS, AGENTS, ACTORS = {}, {}, {}, {}, {}, {}

parser = argparse.ArgumentParser(description="Launch Bray for train or deploy")
parser.add_argument("--config", help="Config yaml file")
parser.add_argument("--trial", help="Override trial name in config file")
args = parser.parse_args()

with open(args.config) as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)

bray.init(project=CONFIG["project"], trial=args.trial or CONFIG["trial"])
bray.set("config", CONFIG)

FILTER_CONFIG = lambda kind: {
    name: conf
    for name, conf in CONFIG.items()
    if isinstance(conf, dict) and conf.get("kind") == kind
}

for name, c in FILTER_CONFIG("model").items():
    model, forward_args = importlib.import_module(c["module"]).build_model()
    remote_model = MODELS[name] = bray.RemoteModel(
        name=name,
        model=model,
        forward_args=forward_args,
        forward_kwargs={},
        checkpoint_interval=c.get("checkpoint_interval"),
        max_batch_size=c.get("max_batch_size"),
        num_workers=c.get("num_workers"),
        cpus_per_worker=c.get("cpus_per_worker"),
        gpus_per_worker=c.get("gpus_per_worker"),
        memory_per_worker=c.get("memory_per_worker"),
        use_onnx=c.get("use_onnx"),
        local_mode=c.get("local_mode"),
        override_model=c.get("override_model"),
    )
    if c.get("tensorboard_graph"):
        bray.add_graph(
            remote_model.get_model().eval(),
            remote_model.get_torch_forward_args(),
        )
    if c.get("tensorboard_step"):
        bray.set_tensorboard_step(remote_model.name)


for name, c in FILTER_CONFIG("source").items():
    sources = importlib.import_module(c["module"]).build_source()
    SOURCES[name] = {
        "sources": sources,
        "num_workers": c.get("num_workers"),
        "epoch": c.get("epoch"),
    }


for name, c in FILTER_CONFIG("buffer").items():
    remote_buffer = BUFFERS[name] = bray.RemoteBuffer(
        name=name,
        size=c.get("size"),
        batch_size=c.get("batch_size"),
        num_workers=c.get("num_workers"),
        density=c.get("density"),
    )
    if not (sources := c.get("sources")):
        continue
    for source in sources:
        remote_buffer.add_source(*SOURCES[source])


for name, c in FILTER_CONFIG("trainer").items():
    remote_trainer = TRAINERS[name] = bray.RemoteTrainer(
        name=name,
        use_gpu=c.get("use_gpu"),
        num_workers=c.get("num_workers"),
        cpus_per_worker=c.get("cpus_per_worker"),
    )
    train = importlib.import_module(c["module"]).train
    remote_trainer.train(train=train)


for name, c in FILTER_CONFIG("agent").items():
    AGENTS[name] = importlib.import_module(c["module"])


for name, c in FILTER_CONFIG("actor").items():
    remote_actor = ACTORS[name] = bray.RemoteActor(
        name=name,
        port=c.get("port"),
        num_workers=c.get("num_workers"),
        actors_per_worker=c.get("actors_per_worker"),
        cpus_per_worker=c.get("cpus_per_worker"),
        memory_per_worker=c.get("memory_per_worker"),
        use_tcp=c.get("use_tcp"),
        use_gateway=c.get("use_gateway"),
    )
    module = importlib.import_module(c["module"])
    Actor = getattr(module, c.get("class") or "Actor")
    remote_actor.serve(Actor=Actor)


for name, c in FILTER_CONFIG("agent_actor").items():
    remote_actor = ACTORS[name] = bray.RemoteActor(
        name=name,
        port=c.get("port"),
        num_workers=c.get("num_workers"),
        actors_per_worker=c.get("actors_per_worker"),
        cpus_per_worker=c.get("cpus_per_worker"),
        memory_per_worker=c.get("memory_per_worker"),
        use_tcp=c.get("use_tcp"),
        use_gateway=c.get("use_gateway"),
    )
    if not (agents := c.get("agents")):
        continue
    remote_actor.serve(
        Actor=bray.AgentActor,
        TickInputProto=c.get("tick_input_proto"),
        TickOutputProto=c.get("tick_output_proto"),
        agents={a: AGENTS[a] for a in agents},
        episode_length=c.get("episode_length"),
    )

print(f"Bray launch success at config {args.config}")
bray.run_until_asked_to_stop()
