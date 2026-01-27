import requests, os, json, shutil, pandas, time
from bray.launch import app, HOST, PORT, uvicorn, executor
from bray.common import (load_trial_config, 
    get_trial_path, get_output_path, get_project_path)
import bray.dataset as d_; import bray.reward as r_
import bray.template as t_; import bray.model as m_
import gradio as gr; import importlib
from datasets import (load_dataset, 
    load_dataset_builder, get_dataset_config_names)

BASE_URL = f'http://127.0.0.1:{PORT}'

TASK_CHOICES = ['GRPO', 'SFT', 'DL', 'RLOO', 'DAPO', 'RM', 
    'DPO', 'KTO', 'PPO', 'RL', 'EVAL', 'NONE', 'RAY']
SERVE_TASKS = ['SERVE', 'API', 'VLLM', 'SGLANG', 'WEB', 'MODEL']
TASK_CHOICES += SERVE_TASKS; SERVE_TASKS = set(SERVE_TASKS)
RL_TASKS = {'GRPO', 'PPO', 'RL', 'RLOO', 'DAPO'}
NO_TRAIN_TASKS = {'NONE', 'RAY', 'EVAL'} | SERVE_TASKS

TASK_HEADERS = ['Task', 'Name', 'Description', 'Type', 'Status']
TASK_WIDTHS = [3, 2, 2, 1, 1]
TRAIN_TYPE_CHOICES = ['Full', 'LoRA']
QUANTIZE_CHOICES = ['None', 'FP4', 'FP8', 'BF16', 'FP16']
BOOST_CHOICES = ['default', 'flashattn', 'unsloth']

AXIS_X_CHOICES = ['step', 'time', 'data']

DEEPSPEED_ZERO_CHOICES = ['Turn-Off', 'Zero-1', 'Zero-2', 
    'Zero-3', 'Offload']
LR_SCHEDULER_CHOICES = ['none', 'linear', 'cosine']

DATASET_HEADERS = ['Name', 'Description', 'Dataset', 'Split']
DATASET_WIDTHS = [1, 2, 5, 1]
KIND_CHOICES = ['ANY', 'TRAIN', 'VAL', 'EVAL', 'TEST', 'ALL']

REWARD_HEADERS = ['Reward', 'Weight', 
    'Model or Func Plugin Path', 'Description']
REWARD_WIDTHS = [3, 1, 5, 3]

LOG_FILTER_CHOICES = ['TAIL-200K', 
    'VERBOSE', 'CLEAN', 'WARNING', 'METRIC', 'HEAD-200K']

EVAL_HEADERS = ['Model API', 'Method', 'Data Srouce',
    'Eval Metric', 'Result']
EVAL_WIDTHS = [2, 1, 2, 2, 1]
EVAL_METHODS = ['HTTP_POST', 'PYTHON', 'HTTP_GET', 'CHAT_BOT']

TRIALS: dict[str: dict[str: dict[str: str]]] = {}
CACHED_TRIAL2CONFIGS: dict[str: dict] = {}
CONDAS = SCRIPTS = REWARDS = MODELS = DATASETS = []

def reload_modules_and_flush_disk(interval: float = 60):
    try: [importlib.reload(m) for m in [d_, r_, m_, t_]]
    except Exception as e: print(f'重新加载失败，请检查文件状态 {e}')
    global CONDAS, SCRIPTS, REWARDS, MODELS, DATASETS
    CONDAS = sorted(os.listdir('./conda'))
    SCRIPTS = [os.path.join(root, f)[2:] for root, _, fs in 
        os.walk('./script') for f in fs]
    REWARDS = [r['name'] for r in r_.REWARDS]
    MODELS = [m['path'] for m in m_.MODELS]
    DATASETS = d_.DATASETS; time.sleep(interval)
    executor.submit(reload_modules_and_flush_disk, interval)

executor.submit(reload_modules_and_flush_disk, interval=60)

def request_json(url: str, method='POST', **kwargs) -> object:
    response = requests.request(method, url, **kwargs)
    if response.status_code == 200: return response.json()
    raise Exception(f'request {url} err: {response.text}')

def initialize_platform(request: gr.Request) -> tuple[dict]:
    trials = request_json(f'{BASE_URL}/dist/task/query', 'GET')
    TRIALS.clear(); TRIALS.update(trials)
    project = request.query_params.get('project', '')
    trial = request.query_params.get('trial', '')
    if task_id := request.query_params.get('task', ''):
       project, trial = task_id.split('/', 1)
    updates = [gr.update() for _ in range(len(PARAMS))]
    for k, v in request.query_params.items():
        if (k := k.upper()) in NAMES: update_param(updates, k, v)
    trials = sorted(TRIALS.get(project, {}))
    return [gr.update(value=project, choices=sorted(TRIALS)
    ), gr.update(value=trial, choices=trials)] + updates

def get_devices(project: str, kind, cpu=False) -> list | dict:
    url = f'{BASE_URL}/dist/device/query?project={project}'
    devices = request_json(url, 'GET')
    filter = lambda x: {h: v[1 if cpu else 0] for h, v in 
        x.items() if v[1 if cpu else 0]}
    if kind is not None: return filter(devices.get(kind, {}))
    devices = {k: filter(d) for k, d in devices.items()}
    return {k: v for k, v in devices.items() if v or not cpu}

def on_project_input(project: str, trial: str=None) -> dict:
    url = f'{BASE_URL}/dist/task/query?project={project}'
    trials = request_json(url, 'GET').get(project, {})
    if project: TRIALS[project] = trials
    update = gr.update(value=trial) if trial else gr.update()
    return update | gr.update(choices=sorted(trials))

def on_train_type_select(train_type: str) -> dict:
    return gr.update(visible=train_type == 'LoRA')

def query_task_status(project: str, trial: str) -> tuple:
    path = f'/dist/task/status?project={project}&trial={trial}'
    return request_json(f'{BASE_URL}{path}', 'GET')

def build_task_type_and_status(task: list) -> list:
    if len(parts := task[0].split('/', 1)) != 2: return task
    if not task[1]: task[1] = parts[1].upper()
    if not task[2]: task[2] = time.strftime('%Y-%m-%d %H:%M:%S')
    status, task_type, _ = query_task_status(parts[0], parts[1])
    status2color = {'RUNNING': 'green', 'SUCCESS': 'blue', 
        'PENDING': 'orange', 'UNKNOWN': 'gray'}
    color = status2color.get(status, 'red')
    template = task[-1].split('template=')[-1].split('&')[0]
    if not template or status != 'UNKNOWN': template = task[0]
    status = f'''<a href="./?template={template}&project={
        parts[0]}&trial={parts[1]}" 
    target="_blank" style="color: {color};">{status}</a>'''
    return task[:3] + [task_type, status]

def update_tasks_type_and_status(tasks: list) -> list | dict:
    tasks_ = [build_task_type_and_status(task) for task in tasks]
    return tasks_ if tasks_ != tasks else gr.update()

def update_task_status(project: str, trial: str) -> tuple:
    on, off = gr.update(visible=True), gr.update(visible=False)
    down = gr.update(interactive=False)
    if not trial: return [down, off, off, down, off, off, down]
    status, task_type, dep = query_task_status(project, trial)
    u = gr.update(value='跳转页面', interactive=True,
         link=f'/{project}/{trial}')
    if task_type != 'WEB' or status != 'RUNNING':
        u = gr.update(value='删除任务', link='') | down
    d = gr.update(visible=True, link=f'?task={dep}')
    launch = down | gr.update(value=status)
    if status in ['RUNNING', 'PENDING']: 
        return [u, d if dep else off, off, launch, on, off, down]
    active = gr.update(interactive=True)
    if status == 'SUCCESS': 
        return [u | active, off, on, launch, off, off, active]
    launch = active | gr.update(value='启动任务')
    if not os.path.exists(get_trial_path(project, trial)): 
        return [u | down, off, on, launch, off, off, down]
    path = get_output_path(project, trial, node=0)
    if not os.path.exists(path): resume = off
    else: resume = on; launch = down | gr.update(value=status)
    return [u | active, off, on, launch, off, resume, active]

def update_param(params: list, name: str, value: object) -> list:
    params[NAMES.index(name)] = value; return params

def get_param_value(name: str, env: dict={}) -> object:
    return env.get(name, DEFAULTS[NAMES.index(name)])

def on_trial_change(template, project, trial) -> tuple:
    trial_path = get_trial_path(project, trial)
    config_path = f'{trial_path}/config.json'
    if '/' in template and not os.path.exists(config_path): 
        project, trial = template.split('/', 1)
    updates = [gr.update() if trial != '' 
        else gr.update(value=default) for default in DEFAULTS]
    if trial == '': update_param(updates, 'TASKS', [
        build_task_type_and_status([f'{project}/{t}', '', '', '']
    ) for t in sorted(TRIALS.get(project, {}))])
    # update trial buttons if trial not exist
    if not os.path.exists(config_path): return updates
    # load trial config from file
    with open(config_path, 'r') as f: env = json.load(f)
    updates = [gr.update(value=env.get(n, DEFAULTS[i])) 
        for i, n in enumerate(PARAMS)]
    cached_config = CACHED_TRIAL2CONFIGS.get(
        f'{project}/{trial}') or (
    CACHED_TRIAL2CONFIGS.get(env.get('DIST_TEMPLATE')) or {})
    for k in NAMES[-6:]: update_param(
        updates, k, cached_config.get(k) or gr.update())
    update_param(updates, 'TASKS', [build_task_type_and_status(
        t + ['', t[-1]]) for t in env.get('TASKS', [])])
    # load env code, eval input/output from file
    def read_text_file_if_exists(path: str) -> str:
        if not os.path.exists(path): return None
        with open(path, 'r') as f: return f.read() or None
    update_param(updates, 'ENV_CODE', 
        read_text_file_if_exists(f'{trial_path}/env.sh'))
    # update eval input/output/label/code from file
    e = read_text_file_if_exists(f'{trial_path}/evals.json')
    if e: update_param(updates, 'EVALS', json.loads(e))
    update_param(updates, 'EVAL_INPUT', 
        read_text_file_if_exists(f'{trial_path}/input.json'))
    update_param(updates, 'EVAL_CODE', 
        read_text_file_if_exists(f'{trial_path}/code.json'))
    # append default datasets to trial
    datasets = env.get('DATASETS', [])
    dataset_paths = [dataset[1] for dataset in datasets]
    datasets += [['', d['desc'], d['path'], ''] for d in 
        DATASETS if d['path'] not in dataset_paths]
    update_param(updates, 'DATASETS', datasets)
    # update device kind value and choices
    device_kind = get_param_value('DIST_DEVICE_KIND', env)
    devices = get_devices(project, kind=None)
    updates = update_param(updates, 'DIST_DEVICE_KIND',
    gr.update(value=device_kind, choices=list(devices)))
    cpu_kind = get_param_value('DIST_CPU_KIND', env)
    devices = get_devices(project, kind=None, cpu=True)
    updates = update_param(updates, 'DIST_CPU_KIND',
    gr.update(value=cpu_kind, choices=list(devices)))
    # update gpu num value and choices
    device_num = get_param_value('DIST_NUM_DEVICES', env)
    choices = on_device_kind_change(project, device_kind)
    updates = update_param(updates, 'DIST_NUM_DEVICES', 
        choices | gr.update(value=device_num))
    cpu_num = get_param_value('DIST_NUM_CPUS', env)
    choices = on_cpu_kind_change(project, cpu_kind)
    updates = update_param(updates, 
        'DIST_NUM_CPUS', choices | gr.update(value=cpu_num))
    # update node num value and choices
    cpu_node_num = get_param_value('DIST_NUM_CPU_NODES', env)
    choices = on_cpu_num_change(project, cpu_kind, cpu_num)
    cpu_node_num = choices | gr.update(value=cpu_node_num)
    update_param(updates, 'DIST_NUM_CPU_NODES', cpu_node_num)

    node_num = get_param_value('DIST_NUM_NODES', env)
    device_cpus = get_param_value('DIST_DEVICE_CPUS', env)
    choices = on_device_num_change(
        project, device_kind, device_num, device_cpus)
    node_num = choices | gr.update(value=node_num)
    return update_param(updates, 'DIST_NUM_NODES', node_num)

def build_trial_config(project, trial, kwargs: dict) -> dict:
    name, template = t_.match_template(**kwargs)
    env = {k: v for k, v in template.items() if k != 'VERIFY' and 
        not isinstance(v, (list, dict))}
    env.update({'TEMPLATE': name})
    kwargs = {k: v for k, v in kwargs.items() if is_task_valid(
        kwargs['DIST_TASK_TYPE'], k) or k in template}

    tasks = [t for t in kwargs.get('TASKS', []) if t[0]]
    for task in tasks: env[task[1]] = task[0]
    if tasks and kwargs['DIST_TASK_DEPS'] == 'ON': 
        env['DIST_TASKS'] = ' '.join([t[0] for t in tasks])

    names = NAMES[:NAMES.index('DIST_DATASET_SPLIT')]
    env.update({k: kwargs[k] for k in names if kwargs.get(k) 
        is not None and k.startswith('DIST_')})

    datasets = [d for d in kwargs.get('DATASETS', []) if d[2]]
    for d in [d for d in datasets if d[0]]: 
        env[d[0]] = d[2]; env[f'{d[0]}_SPLIT'] = d[3]
    if dataset := ' '.join([d[2] for d in datasets if d[0]]):
        env['DIST_DATASET'] = dataset
    kinds = set(k for k in [d[0].split('_DATASET')[0] for 
        d in datasets if d[0]] if k != 'ANY')
    for k in kinds: env[f'{k}_DATASET'] = ' '.join([d[2] for 
        d in datasets if d[0].startswith(k)] )
    
    names = NAMES[len(names):NAMES.index('DIST_NUM_GENS')]
    env.update({k: kwargs[k] for k in names if kwargs.get(k) 
        is not None and k.startswith('DIST_')})
    
    rewards = [r for r in kwargs.get('REWARDS', []) if r[0]]
    if reward_weights := ' '.join([r[1] for r in rewards]):
        env['DIST_REWARD_WEIGHTS'] = reward_weights
    reward_funcs = ' '.join([r[0] for r in rewards])
    if rewards: env['DIST_REWARD_FUNCS'] = reward_funcs
    reward_plugins = ' '.join([
        p for p in set(r[2] for r in rewards if r[2])])
    if reward_plugins: env['DIST_REWARD_PLUGINS'] = reward_plugins

    names = NAMES[NAMES.index('DIST_NUM_GENS'):]
    return env | {k: kwargs[k] for k in names if 
    kwargs.get(k) is not None and k.startswith('DIST_')}

def handle_match_template(updates, name, template) -> list:
    env = {k: v for k, v in template.items() if k in PARAMS}
    update_param(updates, 'TEMPLATE', name)
    for k, v in env.items(): update_param(updates, k, v)

def verify_trial(project: str, trial: str, *args) -> tuple:
    updates = [gr.update() for _ in range(len(PARAMS))]
    kwargs = {n: args[i] for i, n in enumerate(PARAMS)}
    name, template = t_.match_template(**kwargs)
    if name != kwargs['TEMPLATE']: 
        handle_match_template(updates, name, template)
    verify = template.get('VERIFY', lambda **_: None)
    if err_msg := verify(**kwargs): 
        return updates + [f'模版校验失败 {name} {err_msg}']
    model = kwargs['DIST_MODEL']
    CACHED_TRIAL2CONFIGS[f'{project}/{trial}'] = kwargs
    if not model or os.path.exists(model):
        return updates + [f'匹配模版 {name} 成功']
    else: return updates + [f'模型不存在 {model}']

def launch_trial(project, trial, *args, resume=False) -> str:
    if not project or not trial: return '缺失实验名'
    kwargs = {n: args[i] for i, n in enumerate(PARAMS)}
    name, template = t_.match_template(**kwargs)
    verify = template.get('VERIFY', lambda **_: None)
    if err_msg := verify(**kwargs): 
        return f'启动失败 模版校验错误 {name} {err_msg}'
    q = f'?project={project}&trial={trial}&resume={resume}'
    return request_json(f'{BASE_URL}/dist/task/launch{q}')
    
def stop_trial(project: str, trial: str) -> str:
    path = f'/dist/task/stop?project={project}&trial={trial}'
    return request_json(f'{BASE_URL}{path}') or '停止成功'

def resume_trial(*args): return launch_trial(*args, resume=True)

def save_trial(project: str, trial: str, *args) -> tuple:
    if not project or not trial: 
        return [gr.update()] * 3 + ['缺失项目或实验名']
    kwargs = {n: args[i] for i, n in enumerate(PARAMS)}
    config = build_trial_config(project, trial, kwargs)
    if ds := [d for d in kwargs['DATASETS'] if d[0]]: 
        config['DATASETS'] = ds
    if tasks := [t[:-2] + [t[-1].split('template=')[-1].split(
    '&')[0]] for t in kwargs['TASKS'] if t[0]]: 
        config['TASKS'] = tasks
    if r := kwargs['REWARDS']: config['REWARDS'] = r
    trial_path = get_trial_path(project, trial)
    os.makedirs(trial_path, exist_ok=True)
    with open(f'{trial_path}/config.json', 'w') as f: 
        json.dump(config, f, indent=4)
    if (e := kwargs['ENV_CODE']) is not None:
        with open(f'{trial_path}/env.sh', 'w') as f: f.write(e)
    output = f'保存任务 {project}/{trial} 成功'
    trial = on_project_input(project)
    project = gr.update(choices=sorted(TRIALS))
    return project, trial, gr.update(interactive=True), output

def delete_trial(project, trial, delete: str) -> tuple:
    if delete == '跳转页面': return [gr.update()] * 5
    trial_path = get_trial_path(project, trial)
    if not os.path.exists(trial_path):
        return [gr.update()] * 4 + [f'任务 {trial} 不存在']
    on, off = gr.update(visible=True), gr.update(visible=False)
    if delete == '删除任务': 
        return gr.update(), '确认删除', on, off, '请确认'
    url = f'/dist/task/remove?project={project}&trial={trial}'
    if r := request_json(f'{BASE_URL}{url}'):
        return [gr.update()] * 4 + [r]
    if os.path.abspath(trial_path) == trial_path: 
        shutil.rmtree(trial_path)
    else: return [gr.update()] * 4 + [f'任务 {trial} 路径非法']
    update = on_project_input(project) | gr.update(value=None)
    return update, '删除任务', off, on, f'已删除 {trial}'

def on_task_type_change(task_type: str, *args) -> tuple[dict]:
    kwargs = {n: args[i] for i, n in enumerate(PARAMS)}
    updates = [gr.update()] * len(PARAMS)
    on, off = gr.update(visible=True), gr.update(visible=False)
    for k in kwargs: update_param(
        updates, k, on if is_task_valid(task_type, k) else off)
    update_param(updates, 'TASKS', gr.update())
    update_param(updates, 'TEMPLATE', off)
    dataset_group = updates[NAMES.index('DATASETS')]
    if task_type == 'EVAL': dataset_group = on
    reward_group = on if task_type in RL_TASKS else off
    return [dataset_group, reward_group] + updates

def is_task_valid(task_type: str, name: str) -> bool: 
    if task_type not in TASK_CHOICES: task_type = 'NONE'
    return name not in PARAMS or task_type not in PARAMS[name][1]

def on_device_kind_change(project, kind, cpu=False) -> dict:
    devices, choices = get_devices(project, kind, cpu), [1]
    num = max([0] + [len(v) for v in devices.values()])
    while choices[-1] <= num: choices.append(choices[-1] * 2)
    if num and choices[-2] < num: choices.insert(-1, num)
    return gr.update(choices=[str(c) for c in choices[:-1]])

def on_cpu_kind_change(project, kind, device_num: str='') -> dict: 
    return on_device_kind_change(project, kind, cpu=True)

def on_device_num_change(project, device_kind, device_num, 
        device_cpus, cpu=False) -> dict:
    devices, choices = get_devices(project, device_kind, cpu), [1]
    if not device_num: return gr.update(choices=[])
    else: device_num = int(device_num)
    num = len([v for v in devices.values() if len(v) >= device_num])
    while choices[-1] <= num: choices.append(choices[-1] * 2)
    if num and choices[-2] < num: choices.insert(-1, num)
    return gr.update(choices=[str(c) for c in choices[:-1]])

def on_cpu_num_change(*args): 
    return on_device_num_change(*args, device_cpus=None, cpu=True)

def on_reward_select(reward: str, rewards: list) -> list:
    if [r for r in rewards if r[0] == reward]: return rewards
    r = [r for r in r_.REWARDS if r['name'] == reward][0]
    return rewards + [[reward, '1', r['path'], r['desc']]]

def on_ckpt_step_change(project, trial, model, ckpt_step=-1) -> dict:
    path = f'{get_trial_path(project, trial)}/output'
    handle_ckpt = lambda x: int(x.removeprefix('checkpoint-'))
    is_ckpt = lambda x: x.startswith('checkpoint-')
    if not os.path.exists(path): ckpts = [0]
    else: ckpts = sorted([0] + [handle_ckpt(x) for x in 
        os.listdir(path) if is_ckpt(x)])
    step = ckpts[-1] if ckpt_step == -1 else int(ckpt_step)
    diffs = [(c, abs(step - c)) for c in ckpts]
    step = min(diffs, key=lambda x: x[1])[0]
    if step: label = f'{path}/checkpoint-{step}'
    else: label = model or 'Checkpoint Step'
    return gr.update(maximum=ckpts[-1], value=step, label=label)

def get_ckpt_step_path(project: str, trial: str, ckpt_step=-1) -> str:
    return on_ckpt_step_change(project, trial, '', ckpt_step)['label']

def parse_metric_from_json(metric_json, step) -> dict | None:
    metric = json.loads(metric_json.replace("'", '"'))
    if not isinstance(metric, dict): return None
    try: step = int(metric['global_step/max_steps'].split('/')[0])
    except: return metric | {'step': metric.get('step', step)}
    if 'loss' not in metric and 'lm loss' in metric: 
        metric['loss'] = metric['lm loss']
    if 'eval_loss' not in metric and 'eval_lm loss' in metric: 
        metric['eval_loss'] = metric['eval_lm loss']
    time = pandas.Timedelta(metric['elapsed_time'].replace(' ', ''))
    time = time.total_seconds() / 60
    data = step / int(metric['global_step/max_steps'].split('/')[1])
    return metric | {'step': step, 'time': time, 'data': data}

def parse_output_metrics(project: str, trial: str) -> list:
    trial_path = get_trial_path(project, trial)
    metric_path = os.path.join(trial_path, 'output/metric.json')
    if not os.path.exists(metric_path): 
        metric_path = os.path.join(trial_path, 'output/logging.jsonl')
    if not os.path.exists(metric_path): return []
    with open(metric_path, 'r') as f: lines = f.readlines()
    def try_parse_metric(l: str, step: int):
        try: return parse_metric_from_json(l.strip(), step)
        except: return None
    metrics = [try_parse_metric(l, i) for i, l in enumerate(lines)]
    return [metric for metric in metrics if metric]

def flush_log_and_metric(project, trial, metric, log_filter, 
        node=0) -> tuple[dict, dict, dict, str]:
    metrics = parse_output_metrics(project, trial)
    names = sorted({n for m in metrics for n in m})
    if metric not in names and 'reward' in names: metric = 'reward'
    elif metric not in names: metric = 'loss'
    if metric not in names: update = gr.update(choices=names)
    else: update = gr.update(value=metric, choices=names)
    # trials = [t for t in TRIALS.get(project, {}) if t != trial]
    trials = [t for t in TRIALS.get(project, {})]
    metric_label = gr.update(choices=sorted(trials))
    path = get_output_path(project, trial, node)
    if not os.path.exists(path): return update, metric_label, node, ''
    node_path = get_output_path(project, trial, node=1)
    nodes = sorted([int(n.split('.')[1]) for n in os.listdir(
        os.path.dirname(node_path)) 
    if n.startswith('out.') and n.endswith('.txt')])
    resource = request_json(f'{BASE_URL}/dist/task/resource'
        f'?project={project}&trial={trial}', 'GET')
    if r := list(resource.values()): nodes = list(r[0])
    node = gr.update(visible=r or len(nodes) > 1, choices=
        nodes, value=nodes[node or 0])
    hint = 200 * 1024 if log_filter == 'HEAD-200K' else -1
    seek = lambda f: f.seek(max(0, f.seek(0, 2) - 200 * 1024) if 
        log_filter == 'TAIL-200K' else 0)
    with open(path, 'rb') as f: seek(f); data = f.read(hint)
    log = data.decode(errors='replace').split('\n')
    if log_filter not in LOG_FILTER_CHOICES:
        log = [l for l in log if log_filter in l]
    if log_filter == 'WARNING': 
        log = [l for l in log if 'WARN' in l or 'ERROR' in l]
    IGNORES = ['INFO', 'WARN', 'deprecated']
    filter = lambda l: all(i not in l for i in IGNORES)
    if log_filter == 'CLEAN': log = [l for l in log if filter(l)]
    if log_filter == 'METRIC': 
        log = ['\n'.join(str(m) for m in metrics)]
    return update, metric_label, node, '\n'.join(log).strip()

def on_metric_select(project, trial, metric, metric_label, axis_x):
    metrics = parse_output_metrics(project, trial)
    metrics = [m for m in metrics if metric in m]
    for i in range(len(metrics) - 2, -1, -1): 
        if metrics[i]['step'] < metrics[i + 1]['step']: continue
        metrics[i] = metrics[i + 1]
    x, y = [m[axis_x] for m in metrics], [m[metric] for m in metrics]
    if trial in metric_label: metric_label.remove(trial)
    if not metric_label: updates = {}
    else: updates = on_metric_select(
        project, metric_label[0], metric, metric_label[1:], axis_x)
    data = updates['value'].to_dict(orient='list') if updates else {}
    t = [trial] * len(metrics) + data.get('trial', [])
    x += data.get('x', []); y += data.get('y', [])
    color = f'color-{str(id(updates))[-4:]}'
    value = pandas.DataFrame({'x': x, 'y': y, 'trial': t, color: t})
    return gr.update(title=metric, y_title=metric, 
    color=color if t else None, x_title=axis_x, value=value)

def clean_log(project, trial, clean, node=0) -> str:
    if clean == '清理日志': return '确认清理', '请确认清理日志'
    path = f'{get_trial_path(project, trial)}/output'
    if (os.path.abspath(path) != path or not os.path.exists(path)): 
        return '清理日志', '日志文件不存在'
    url = f'/dist/task/remove?project={project}&trial={trial}'
    if r := request_json(f'{BASE_URL}{url}'): return '清理日志', r
    shutil.rmtree(path); return '清理日志', '日志已清理'

def on_eval_start(project, trial, api, method, input):
    base_url = f'http://localhost:8413/{project}/{trial}'
    if not api or not api.startswith('http'):
        api = f'{base_url}{api}'
    method = 'POST' if method == 'HTTP_POST' else 'GET'
    data = requests.request(method, api, data=input.encode('utf-8'), 
    headers={'Content-Type': 'application/json'}).json()
    return json.dumps(data, indent=4, ensure_ascii=False)

def on_script_input(script, code) -> tuple[dict, float]:
    path = os.path.join(code or './', script)
    if script.startswith('script/') and not os.path.exists(path):
        path = os.path.join('./', script)
    updates = gr.update(value=None, label=script, visible=True)
    if not os.path.isfile(path): return updates, 0.0
    with open(path, 'r') as f: code = f.read()
    language = 'python' if script.endswith('.py') else 'shell'
    updates |= gr.update(value=code, language=language)
    return updates, os.path.getmtime(path)

def on_conda_input(conda: str) -> tuple[dict, float]:
    if not os.path.isfile(path := os.path.join('conda', conda)):
        return gr.update(label=conda, visible=True), 0.0
    with open(path, 'r') as f: code = f.read()
    updates = gr.update(value=code, label=conda, visible=True)
    return updates, os.path.getmtime(path)

def on_record_click(project, record, rec_md) -> dict | str:
    if record == '实验记录' and rec_md: return gr.update()
    path = f'{get_project_path(project)}/record.md'
    if record == '实验记录' and os.path.exists(path): 
        with open(path, 'r') as f: return f.read()
    with open(path, 'w') as f: f.write(rec_md); return gr.update()

def save_conda_code(conda, conda_code, mtime) -> dict:
    if not conda or not conda_code or mtime > 0: return gr.update()
    path = os.path.join('conda', conda)
    if os.path.exists(path) and os.path.getmtime(path) > abs(mtime):
        raise gr.Error('Conda文件被外部修改，请刷新页面')
    with open(path, 'w') as f: f.write(conda_code)
    global CONDAS; CONDAS = sorted(os.listdir('./conda'))
    return gr.update(value=max(os.path.getmtime(path), abs(mtime)))

def save_script_code(code, script, script_code, mtime) -> dict:
    if not script or not script_code or mtime > 0: return gr.update()
    path = os.path.join(code or './', script)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getmtime(path) > abs(mtime):
        raise gr.Error('脚本文件被外部修改，请刷新页面')
    with open(path, 'w') as f: f.write(script_code)
    return gr.update(value=max(os.path.getmtime(path), abs(mtime)))

def build_config(project: str, trial: str, *args) -> str:
    kwargs = {n: args[i] for i, n in enumerate(PARAMS)}
    config = build_trial_config(project, trial, kwargs)
    extra_env = {'DIST_MASTER': 'ip', 'DIST_MASTER_PORT': 'int', 
    'DIST_WORKER': 'ip', 'DIST_NODE_RANK': 'int', 'DIST_RESUME_PATH': '', 
    'DIST_TRIAL': trial, 'DIST_PROJECT': project}
    extra_env['DIST_TRIAL_PATH'] = get_trial_path(project, trial)
    return json.dumps(config | extra_env, indent=0)

def on_code_or_script_change(code: str, script: str) -> dict:
    if code != './' and script.startswith('script/'): 
        defaults = on_code_or_script_change('./', script)
    else: defaults = gr.update(choices=[])
    if not script: defaults['choices'].append('script/')
    if not code: return gr.update(choices=SCRIPTS)
    if not script.endswith('/'): script = os.path.dirname(script)
    path = os.path.join(code, script)
    if not os.path.isdir(path): return defaults
    choices = [os.path.join(script, d + '/' if os.path.isdir(
        os.path.join(path, d)) else d) for d in os.listdir(path)]
    return gr.update(choices=choices + [
    c for c in defaults['choices'] if c not in choices])

def on_script_key_up(code: str, data: gr.KeyUpData):
    return on_code_or_script_change(code, data.input_value)
    
def build_execute_group(project, trial, saves) -> tuple:
    with gr.Row(equal_height=True) as execute_row:
        conda = gr.Dropdown(allow_custom_value=True, value='',
        label='Conda', scale=3, min_width=100)
        conda_scope = gr.Dropdown(['global', 'trial'], min_width=80, 
        scale=1, label='Scope', visible=False)
    with execute_row as code_row:
        code = gr.Dropdown(allow_custom_value=True, 
        scale=3, label='Code', min_width=100)
    with execute_row as script_row:
        script = gr.Dropdown(allow_custom_value=True, value='', 
        label='Script', scale=3, min_width=100)
        script_scope = gr.Dropdown(['code', 'trial'], min_width=80, 
        scale=1, label='Scope', visible=False)
    conda_row = gr.Row(visible=False)
    with conda_row, gr.Column(min_width=240) as conda_column:
        docker_image = gr.Dropdown(
        allow_custom_value=True, label='Docker Image')
    with conda_row, conda_column, gr.Row(equal_height=True):
        user_group = gr.Dropdown(
        allow_custom_value=True, label='User Group', scale=2)
        user = gr.Textbox(label='User', scale=2)
        resource_group = gr.Dropdown(
        allow_custom_value=True, label='Resource Group', scale=3)
        node_alive = gr.Number(
        600, label='Node Alive', scale=1, min_width=100)
    conda_mtime = gr.Number(0.0, visible=False)
    with conda_row, conda_column as resource_column:
        resource_cfg = gr.Code(label='Resource Config')
    with conda_row as conda_code_row:
        conda_code = gr.Code(language='shell', label='conda.sh')
    script_row = gr.Row(visible=False)
    with script_row, gr.Column(min_width=240) as script_column:
        script_args = gr.Textbox(label='Script Args')
        env_code = gr.Code(language='shell', label='env.sh')
    with script_row, script_column:
        script_cfg = gr.Code(language='json', label='config.json')
    with script_row: script_code = gr.Code(language='shell')
    script_mtime = gr.Number(0.0, visible=False)
    on, off = gr.update(visible=True), gr.update(visible=False)
    code.focus(lambda: (off, off), None, [conda_row, script_row])
    code.blur(lambda: [on] * 2, None, [conda, script])
    code.focus(lambda: [off, on, off], None, 
        [conda, code, script], show_progress='hidden')
    conda.focus(lambda: (on, off), None, [conda_row, script_row])
    conda.change(on_conda_input, conda, [conda_code, conda_mtime])
    script.change(on_script_input, [script, code], 
        [script_code, script_mtime])
    script.focus(lambda: (on, off), None, [script_row, conda_row])
    for c in saves: c.click(save_script_code, 
        [code, script, script_code, script_mtime], script_mtime)
    script_code.input(lambda x: -abs(x), script_mtime, script_mtime)
    for c in saves: c.click(save_conda_code, 
        [conda, conda_code, conda_mtime], conda_mtime
    ).then(lambda: gr.update(choices=CONDAS), None, conda)
    conda_code.input(lambda x: -abs(x), conda_mtime, conda_mtime)
    code.change(on_code_or_script_change, 
        [code, script], script).then(
    on_script_input, [script, code], [script_code, script_mtime])
    script.input(on_code_or_script_change, [code, script], script)
    script.key_up(on_script_key_up, 
        [code], script, show_progress='hidden')
    resource = [user, user_group, resource_group, node_alive]
    return (conda, code, script, script_args, script_cfg, 
    docker_image, *resource, resource_cfg, env_code)

def on_kind_select(kind, dataset, datasets) -> tuple:
    ds = [d.copy() for d in datasets if kind == 'ANY' and d[0]
        or kind in d[0] or kind == 'ALL']
    dataset_choices = gr.update(choices=[d[2] for d in ds])
    return dataset_choices | gr.update(value=dataset), ds

def on_datasets_input(kind, ds: list, datasets: list) -> list:
    _, filter_ds = on_kind_select(kind, '', datasets)
    return [d for d in datasets if d not in filter_ds] + ds

def add_dataset(kind, dataset, ds, datasets) -> tuple:
    if not dataset: return [gr.update()] * 2 + ['数据集为空']
    _, filter_ds = on_kind_select(kind, '', datasets)
    d = [d for d in filter_ds if d[2] == dataset]
    if d: return [gr.update()] * 2 + [f'数据集已存在 {dataset}']
    if not os.path.exists(dataset):
        return [gr.update()] * 2 + [f'数据集不存在 {dataset} ']
    postfix = '' if not filter_ds else f'_{len(filter_ds)}'
    d = [f'{kind}_DATASET{postfix}', '', dataset, '']
    return ds + [d], datasets + [d.copy()], gr.update()

def preview_dataset(preview: str) -> tuple[str, dict, dict]:
    ds, column = gr.update(visible=True), gr.update(visible=False)
    if preview == '预览': ds, column = column, ds
    return '完成' if preview == '预览' else '预览', ds, column

def build_preview_ds(dataset, page_size, page) -> dict:
    data = dataset[page_size * (page - 1): page_size * page]
    avg_len = lambda v: sum(map(lambda x: len(x), v)) / len(v)
    data = {k: [str(x) for x in v] for k, v in data.items()}
    lens = [int(avg_len(v)) for v in data.values() if v]
    column_widths = [f'{10*min(52, max(5, l)+1)}px' for l in lens]
    value = pandas.DataFrame(data)
    return gr.update(value=value, column_widths=column_widths)

def load_dataset_(ds_path, name, split, file) -> 'Dataset':
    if not file: return load_dataset(ds_path, name, split=split)
    file = ds_path if file is True else os.path.join(ds_path, file)
    return load_dataset(os.path.dirname(file), data_files=file)

def on_preview_change(preview, dataset, name, split, diff, 
        file, filter, page_size, page, ds=None) -> tuple:
    updates = [gr.update()] * 5 + [gr.update(visible=False)] * 4
    if preview == '预览' or not dataset: return updates
    if not os.path.exists(dataset): raise gr.Error('数据集不存在') 
    if os.path.isfile(ds_path := dataset): file = True
    names = [] if file else get_dataset_config_names(ds_path)
    name = None if not names else name if name in names else (
        'default' if 'default' in names else names[0])
    splits = [] if file else list(load_dataset_builder(
        ds_path, name).info.splits or [])
    split = split or splits[0] if splits else None
    try: dataset = load_dataset_(ds_path, name, split, file)
    except: split = filter = None; dataset = []
    if (file or not split) and 'train' in dataset: 
        dataset = dataset['train']
    origin_dataset, data = dataset, gr.update(visible=True)
    if not diff: dd = gr.update(visible=False)
    else: ds, dd = on_preview_change(preview, ds_path, name, split, 
        None, diff, filter, page_size, page, dataset)
    # if ds: dataset = dataset.filter(lambda x, idx: 
    #     idx >= len(ds) or x != ds[idx], with_indices=True)
    if filter: dataset = dataset.filter(
        lambda x: any(filter in str(v) for v in x.values()))
    page_size = int(page_size.split('/')[0])
    minimum, maximum = 1, len(dataset) // page_size
    if len(dataset) % page_size: maximum += 1
    page = min(max(page, 1), maximum)
    try: data |= build_preview_ds(dataset, page_size, page)
    except: data |= gr.update(value=[])
    if not diff and ds: return origin_dataset, data
    page = gr.update(value=page, maximum=maximum, 
        minimum=min(minimum, maximum - 1))
    split = gr.update(visible=not file, value=split, choices=splits)
    name = gr.update(visible=not file, value=name, choices=names)
    markdown = config = ''
    if os.path.exists(path := f'{ds_path}/README.md'):
        with open(path, 'r') as f: markdown = f.read()
    if markdown.startswith('---'): 
        _, config, markdown = markdown.split('---', 2)
    md = gr.update(visible=bool(markdown and not file))
    cfg = gr.update(visible=bool(config and not file))
    files = [os.path.relpath(os.path.join(root, f), ds_path) 
        for root, _, fs in os.walk(
    ds_path, followlinks=True) for f in fs if '.git' not in f]
    diff = gr.update(visible=bool(file), choices=files)
    file = diff | gr.update(visible=file != True)
    md |= gr.update(value=markdown); cfg |= gr.update(value=config)
    return name, split, diff, file, page, dd, data, md, cfg

def build_dataset_preview(dataset, preview, ds):
    with gr.Column(visible=False) as column: pass
    with column, gr.Row(equal_height=True) as operate_row:
        name = gr.Dropdown(label='Config Name', scale=2)
    with column, operate_row as split_row:
        split = gr.Dropdown(
        label='Split', scale=2, allow_custom_value=True)
    with column, operate_row as diff_row:
        diff = gr.Dropdown(label='Diff', 
        allow_custom_value=True, scale=2, visible=False)
    with column, operate_row as filter_row:
        filter = gr.Textbox(label='Filter', scale=3)
    with column, operate_row as path_row:
        file = gr.Dropdown(
        label='File', allow_custom_value=True, scale=2)
    with column, operate_row as page_size_row: 
        page_size = gr.Dropdown(
        [f'{i}/Page' for i in [10, 20, 50, 100]], scale=1,
        show_label=False, min_width=100)
    with column, operate_row as page_row: 
        page = gr.Slider(label='Page', scale=2)
    with column, gr.Row(equal_height=True) as diff_row:
        diff_data = gr.Dataframe(type='array', 
        interactive=True, show_row_numbers=True, visible=False)
    with column, diff_row as data_row:
        data = gr.Dataframe(type='array', 
        interactive=True, show_row_numbers=True)
    with column, gr.Row(equal_height=True) as markdown_row: pass
    with column, markdown_row, gr.Column(scale=2):
        markdown = gr.Markdown(container=True)
    with column, markdown_row, gr.Column(scale=1):
        config = gr.Code(language='yaml', show_label=False)
    on_preview_change_event_args = (on_preview_change, 
        [preview, dataset, name, split, diff, file, 
        filter, page_size, page], [name, split, 
    diff, file, page, diff_data, data, markdown, config])
    for c in [name, split, diff, file, page_size, page]:
        c.input(*on_preview_change_event_args)
    preview.click(preview_dataset, preview, [preview, ds, column]
    ).then(*on_preview_change_event_args)
    dataset.change(lambda: [None] * 5, None, [name, split, 
        diff, file, page]).then(*on_preview_change_event_args)
    filter.submit(*on_preview_change_event_args)

def build_dataset_group(output: gr.Textbox) -> tuple[gr.Blocks]:
    with gr.Group(visible=True) as group: pass
    with group, gr.Row(equal_height=True) as data_row:
        kind = gr.Dropdown(KIND_CHOICES, scale=1, 
        label='Kind', min_width=100)
    with group, data_row as dataset_row:
        dataset = gr.Dropdown(label='Dataset or Path', scale=9, 
        allow_custom_value=True)
    with group, data_row, gr.Column(scale=1, min_width=100):
        dataset_split = gr.Number(0.01, step=0.01, 
        minimum=0, maximum=None, label='Split')
    with group, data_row, gr.Column(scale=1, min_width=100):
        add = gr.Button(value='添加')
        preview = gr.Button(value='预览')
    with group as dataset_group: 
        ds = gr.Dataframe(column_widths=DATASET_WIDTHS,
        headers=DATASET_HEADERS, type='array')
    with group, gr.Row(visible=False):
        datasets = gr.Dataframe(type='array', visible=False)
    init_event_args = (on_kind_select, 
        [kind, dataset, datasets], [dataset, ds])
    add.click(add_dataset, [kind, dataset, ds, datasets], 
        [ds, datasets, output]).then(*init_event_args)
    with group as preview_group: 
        build_dataset_preview(dataset, preview, ds)
    for e in [kind.change, kind.focus]: e(*init_event_args)
    ds.change(on_datasets_input, [kind, ds, datasets], datasets)
    return init_event_args, dataset, dataset_split, datasets

def on_data_source_change(data_source: str) -> tuple[dict]:
    on, off = gr.update(visible=True), gr.update(visible=False)
    eval_input = on if not data_source else off
    if not data_source: eval_output, output_dataset = on, off
    else: eval_output, output_dataset = off, on
    return eval_input, eval_output, output_dataset

def add_eval(project, trial, evals, model_api, eval_method, 
        data_source, eval_metric) -> dict:
    e = [model_api, eval_method, data_source, eval_metric]
    if e in [e[:-1] for e in evals]: return gr.update()
    new_evals = evals + [e + ['']]
    save_eval(project, trial, new_evals, None, None, None)
    return gr.update(value=new_evals, visible=True)

def remove_eval(project, trial, evals, model_api, eval_method, 
        data_source, eval_metric) -> dict:
    e = [model_api, eval_method, data_source, eval_metric]
    new_evals = [ee for ee in evals if e != ee[:-1]]
    if len(new_evals) == evals: return gr.update()
    save_eval(project, trial, new_evals, None, None, None)
    return gr.update(value=new_evals, visible=bool(new_evals))

def save_eval(project, trial, evals, eval_input, 
        eval_code, eval_metric='', metric_code=None, mtime=0.0):
    trial_path = get_trial_path(project, trial)
    os.makedirs(trial_path, exist_ok=True)
    def write_text_file_if_needed(path, content, force=True):
        if content is None: return
        if not force and os.path.exists(path): return
        with open(path, 'w') as f: return f.write(content)
    evals_path = f'{trial_path}/evals.json'
    if evals: write_text_file_if_needed(evals_path, 
        json.dumps(evals, indent=4))
    elif os.path.exists(evals_path): os.remove(evals_path)
    write_text_file_if_needed(
        f'{trial_path}/input.json', eval_input, mtime != 0)
    write_text_file_if_needed(
        f'{trial_path}/eval.py', eval_code, mtime != 0)
    if not eval_metric or mtime >= -1.0: return 0.0
    path = os.path.join(get_project_path(project), eval_metric)
    if os.path.exists(path) and os.path.getmtime(path) > abs(mtime):
        raise gr.Error('指标代码文件被外部修改，请刷新页面')
    write_text_file_if_needed(path, metric_code)
    return max(os.path.getmtime(path), abs(mtime))

def on_eval_info_change(evals, model_api, eval_method, 
        data_source, eval_metric, override=False) -> tuple[dict]:
    on, off = gr.update(visible=True), gr.update(visible=False)
    update = gr.update(value=evals, visible=bool(evals))
    e = [model_api, eval_method, data_source, eval_metric]
    exist = e in [e[:-1] for e in evals]
    if not exist and override and evals: e = evals[0]
    if exist or (override and evals): remove, add = on, off
    else: remove, add = off, on
    choices = list(set([e[0] for e in evals]))
    model_api = gr.update(value=e[0], choices=choices)
    eval_method = gr.update(value=e[1])
    choices = list(set([e[2] for e in evals if e[2]]))
    data_source = gr.update(value=e[2], choices=choices)
    defaults = ['has_diff.py']
    choices = defaults + list(set([e[3] for e in evals 
        if e[3] not in defaults]))
    eval_metric = eval_metric or e[3] or defaults[0]
    eval_metric = gr.update(value=eval_metric, choices=choices)
    updates = [model_api, eval_method, data_source, eval_metric]
    return [update] + updates + [remove, add]

def on_eval_btn_click(*args): return on_eval_info_change(*args, True)

def on_eval_metric_change(project, eval_metric) -> tuple:
    updates = gr.update(label=eval_metric, language='python')
    path = os.path.join(get_project_path(project), eval_metric)
    if not os.path.exists(path):
        path = os.path.join('./metric/', eval_metric)
    if not os.path.isfile(path): return updates, -2.0
    with open(path, 'r') as f: code = f.read()
    return updates | gr.update(value=code), os.path.getmtime(path)

def build_eval_group(project, trial, eval_btn, saves):
    evals = gr.Dataframe(headers=EVAL_HEADERS, type='array', 
        column_widths=EVAL_WIDTHS, visible=False)
    with gr.Row(equal_height=True) as code_row:
        eval_code = gr.Code(
        language='python', label='eval.py', visible=False)
        metric_code = gr.Code(language='python', visible=False)
    metric_code_mtime = gr.Number(0.0, visible=False)
    with gr.Row(equal_height=True) as eval_row:
        model_api = gr.Dropdown(
        allow_custom_value=True, label='Model API', scale=2)
    with eval_row as eval_method_row:
        eval_method = gr.Dropdown(EVAL_METHODS, 
        allow_custom_value=True, label='Method', scale=1)
    with eval_row as eval_data_source_row:
        data_source = gr.Dropdown(value='',
        allow_custom_value=True, label='Data Source', scale=2)
    with eval_row as eval_metric_row:
        eval_metric = gr.Dropdown(value='',
        allow_custom_value=True, label='Eval Metric', scale=2)
    with eval_row, gr.Column(scale=1, min_width=100):
        remove = gr.Button('移除', visible=False)
        add, eval = gr.Button('添加'), gr.Button('评估')
        eval_save = gr.Button('保存', visible=False)
    with gr.Row(equal_height=True) as output_row:
        eval_input = gr.Code(language='json', label='input.json')
        eval_output = gr.Code(
        language='json', label='output.json')
    with gr.Row(equal_height=True, visible=False) as ds_row:
        output_ds = gr.Textbox(label='Output', scale=7)
    with ds_row, gr.Column(scale=1, min_width=100):
        clean, preview = gr.Button('清理'), gr.Button('预览')
    with gr.Column(visible=False) as chat_column:
        chatbot = gr.Chatbot()
        chat_input = gr.MultimodalTextbox(
        interactive=True, show_label=False, file_count='multiple', 
        placeholder='Enter message or upload file...',
        sources=['microphone', 'upload'])
    eval_info = [model_api, eval_method, data_source, eval_metric]
    eval_method.change(lambda x: gr.update(visible=x=='CHAT_BOT'), 
        eval_method, chat_column)
    on, off = gr.update(visible=True), gr.update(visible=False)
    data_source_event_args = (lambda x: (on, off) if x else (off, on), 
        data_source, [ds_row, output_row])
    for c in [model_api, data_source]: c.focus(
        lambda: off, None, code_row)
    data_source.change(*data_source_event_args)
    eval_metric.change(on_eval_metric_change, [project, eval_metric], 
        [metric_code, metric_code_mtime])
    for c in saves + [eval_save]: c.click(save_eval, 
        [project, trial, evals, eval_input, eval_code, 
    eval_metric, metric_code, metric_code_mtime], metric_code_mtime
    ).success(lambda: (on, off), None, [eval, eval_save])
    for c in [eval_input, eval_code, metric_code]: 
        c.input(lambda x: (off, on, -abs(x) or -1), metric_code_mtime, 
        [eval, eval_save, metric_code_mtime])
    add_remove_eval_args = [project, trial, evals] + eval_info
    add.click(add_eval, add_remove_eval_args, evals)
    remove.click(remove_eval, add_remove_eval_args, evals)
    for c in [eval_method, eval_metric]: c.focus(
        lambda: on, None, code_row)
    eval.click(on_eval_start, [project, trial, model_api, 
        eval_method, eval_input], eval_output)
    for e in [evals.change, eval_btn.click]: e(on_eval_btn_click, 
        [evals] + eval_info, [evals] + eval_info + [remove, add]
    ).then(lambda: None, None, eval_output)
    build_dataset_preview(output_ds, preview, evals)
    for c in eval_info: c.change(on_eval_info_change, 
        [evals] + eval_info, [evals] + eval_info + [remove, add])
    return evals, eval_input, eval_code, eval_info
    # chat_msg = chat_input.submit(lambda x, y: (x, y), 
    #     [chatbot, chat_input], [chatbot, chat_input])
    # with eval_group, gr.Row(equal_height=True) as eval_row:
    #     eval_code = gr.Code(language='python', label='eval.py')
    # bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name='bot_response')
    # bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

with gr.Blocks(title='Bray Cloud') as platform:
    style, title = 'style="text-align:center"', 'Bray Cloud'
    gr.Markdown(f'<h1 {style}>{title}</h1>')
    state, template = gr.State({}), gr.Textbox('', visible=False)
    with gr.Row(equal_height=True) as trial_row:
        project = gr.Dropdown(label='项目名', min_width=200,
        scale=2, allow_custom_value=True)
        trial = gr.Dropdown(label='实验名', min_width=300,
        scale=3, allow_custom_value=True)
    with trial_row as output_row:
        output = gr.Textbox(label='执行结果', scale=2, min_width=160, 
        interactive=False, max_lines=1)
    with trial_row, gr.Column(scale=1, min_width=100):
        delete = gr.Button(value='删除任务', interactive=False)
        cancel = gr.Button(value='取消删除', visible=False)
        dependent = gr.Button(value='跳转依赖', visible=False)
        save = gr.Button(value='暂存任务', visible=False)
    with trial_row, gr.Column(scale=1, min_width=100):
        launch = gr.Button(value='启动任务', interactive=False)
        stop = gr.Button(value='停止任务', visible=False)
        resume = gr.Button(value='继续任务', visible=False)
    saves = [save, launch, resume]
    tasks = gr.Dataframe(column_widths=TASK_WIDTHS, visible=False, 
        headers=TASK_HEADERS, datatype='html', type='array')
    with gr.Row(equal_height=True) as task_row:
        task_deps = gr.Dropdown(label='Task Deps', scale=1,
        choices=['OFF', 'ON'], min_width=100)
    with task_row as task_type_row:
        task_type = gr.Dropdown(label='Task Type', scale=1,
        choices=TASK_CHOICES, allow_custom_value=True, min_width=100)
    with task_row as device_resource_row:
        device_kind = gr.Dropdown(label='Device Kind', scale=2,
        allow_custom_value=True, min_width=100)
        device_num = gr.Dropdown(label='Num Devices', 
        allow_custom_value=True, scale=1, min_width=100)
        device_cpus = gr.Dropdown(label='Num CPUs', 
        allow_custom_value=True, scale=1, min_width=100)
        device_memory = gr.Dropdown(label='Memory/GB', 
        allow_custom_value=True, scale=1, min_width=100)
        node_num = gr.Dropdown(label='Num Nodes', 
        allow_custom_value=True, scale=1, min_width=100)
    with task_row as cpu_resource_row:
        cpu_kind = gr.Dropdown(label='CPU Kind', scale=2,
        allow_custom_value=True, visible=False, min_width=100)
        cpu_num = gr.Dropdown(label='Num CPUs', visible=False,
        allow_custom_value=True, scale=1, min_width=100)
        cpu_memory = gr.Dropdown(label='Memory/GB', visible=False,
        allow_custom_value=True, scale=1, min_width=100)
        cpu_node_num = gr.Dropdown(label='Num Nodes', visible=False,
        allow_custom_value=True, scale=1, min_width=100)
    with task_row as instance_row:
        instance_num = gr.Number(1, label='Num Service', 
        scale=1, minimum=0, visible=False, min_width=120)
    with gr.Row(equal_height=True) as model_row:
        model = gr.Dropdown(MODELS, label='Model or Path', 
        allow_custom_value=True, scale=1, min_width=240)
    with model_row: algo_coloumn = gr.Column(scale=6)
    with algo_coloumn, gr.Row() as algo_row:
        deepspeed_stage = gr.Dropdown(label='DeepSpeed', 
        scale=1, choices=DEEPSPEED_ZERO_CHOICES, min_width=100)
    with algo_row as context_paralle_row:
        cp_size = gr.Dropdown(label='CP Size', 
        allow_custom_value=True, scale=1, min_width=80)
    with algo_row as ep_tp_pp_row:
        ep_size = gr.Dropdown(label='EP Size', 
        allow_custom_value=True, scale=1, min_width=80)
        tp_size = gr.Dropdown(label='TP Size', 
        allow_custom_value=True, scale=1, min_width=80)
        pp_size = gr.Dropdown(label='PP Size', 
        allow_custom_value=True, scale=1, min_width=80)
    with algo_row as quant_row:
        quantize = gr.Dropdown(QUANTIZE_CHOICES, scale=1, 
        label='Quantization', min_width=100)
        train_type = gr.Dropdown(TRAIN_TYPE_CHOICES, scale=1, 
        label='Tuning Method', min_width=120)
        boost = gr.Dropdown(BOOST_CHOICES, 
        scale=1, label='Booster Method', min_width=120)
    with gr.Row(equal_height=True, visible=False) as lora_row:
        lora_rank = gr.Number(8, label='LoRA Rank')
        lora_alpha = gr.Number(16, label='LoRA Alpha')
        lora_dropout = gr.Number(
        minimum=0, maximum=1, step=0.01, label='LoRA Dropout')
    with lora_row as lora_module_row:
        lora_module = gr.Textbox(label='LoRA modules', 
        placeholder='Default or use commas to separate')
    with gr.Group() as dataset_group:
        (init_ds_event_args, dataset, dataset_split, datasets
        ) = build_dataset_group(output)
    with gr.Row(equal_height=True) as prompt_row:
        chat_template = gr.Dropdown(
        allow_custom_value=True, label='Chat Template', scale=2)
        system_prompt = gr.Textbox('You are a helpful assistant.',
        label='System Prompt', scale=10)
        max_seq_len = gr.Number(1024, 
        label='Max Seq Len', scale=1, minimum=0, min_width=100)
        max_gen_len = gr.Number(1024, 
        label='Max Gen Len', scale=1, minimum=0, min_width=100)
    with gr.Row() as train_and_eval_row:
        batch_size = gr.Number(1, label='Batch Size')
    with train_and_eval_row as step_row:
        epoch = gr.Number(1, label='Epoch', minimum=1, min_width=80)
        grad_accum = gr.Number(1,
        label='Grad Accum', minimum=1, min_width=100)
        grad_clip = gr.Number(1.0, label='Grad Clip', min_width=80)
        lr_rate = gr.Number(
        placeholder='1e-5', label='LR Rate', min_width=80)
    with train_and_eval_row as eval_save_row:
        val_step = gr.Number(50, label='Val Step', min_width=80)
        save_step = gr.Number(50, label='Save Step', min_width=80)
        save_limit = gr.Number(
        3, label='Save Limit', minimum=0, min_width=100)
        log_step = gr.Number(1, label='Log Step', min_width=80)
    with train_and_eval_row as gradient_row:
        lr_scheduler = gr.Dropdown(
        label='LR Scheduelr', choices=LR_SCHEDULER_CHOICES)
    reward_group = gr.Group(visible=True)
    with reward_group, gr.Row(equal_height=True) as reward_row:
        reward = gr.Dropdown(REWARDS, scale=1, 
        label='Reward', value=None)
    with reward_row as generate_row:
        gen_num = gr.Number(value=8, label='Num Gens', scale=1)
        gen_temp = gr.Number(value=0.9, label='Gen Temp', 
        scale=1, minimum=0, maximum=1)
        top_p = gr.Number(value=0.9, label='Top P', scale=1, 
        minimum=0, maximum=1)
        top_k = gr.Number(50, label='Top K', scale=1)
    with reward_group: rewards = gr.Dataframe(type='array',
        headers=REWARD_HEADERS, column_widths=REWARD_WIDTHS)
    with gr.Group() as execute_group:
        (conda, code, script, script_args, script_cfg, 
        docker_image, user, user_group, resource_group, 
        node_alive, resource_cfg, env_code,
        ) = build_execute_group(project, trial, saves)
    with gr.Row(equal_height=True) as operate_row:
        node_btn = gr.Button('节点', elem_id='node', min_width=70)
        export_btn = gr.Button(
        '导出', elem_id='export', min_width=70)
        eval_btn = gr.Button('评估', elem_id='eval', min_width=70)
        record_btn = gr.Button(
        '记录', elem_id='record', min_width=70)
        code_btn = gr.Button('代码', elem_id='code', min_width=70)
        schedule_btn = gr.Button(
        '调度', elem_id='schedule', min_width=70)
        file_btn = gr.Button('文件', elem_id='file', min_width=70)
        monitor_btn = gr.Button(
        '监控', elem_id='monitor', min_width=70)
        tb_btn = gr.Button('指标', elem_id='tb', min_width=70)
        log_btn = gr.Button('日志', elem_id='log', min_width=70)
    ckpt_step = gr.Slider(value=-1, visible=False,
        minimum=0, maximum=0, step=1, label='Checkpoint Step')
    with gr.Group(visible=True) as log_group: plot = gr.LinePlot(
        x='x', y='y', title='Metric', height=320)
    with log_group, gr.Row(equal_height=True) as log_row:
        metric = gr.Dropdown(label='Metric', 
        allow_custom_value=True, scale=2, min_width=100)
        axis_x = gr.Dropdown(AXIS_X_CHOICES, label='Axis-X', 
        allow_custom_value=True, scale=1, min_width=100)
        metric_label = gr.Dropdown(label='Metric Label', scale=3, 
        multiselect=True, allow_custom_value=True)
    with log_group, log_row:
        log_filter = gr.Dropdown(LOG_FILTER_CHOICES, scale=1,
        label='Log Filter', allow_custom_value=True, min_width=100)
    with log_group, log_row:
        node = gr.Dropdown([0], label='Node', scale=1, 
        min_width=60, visible=False, type='index')
    with log_group, log_row, gr.Column(scale=1, min_width=100):
        clean = gr.Button('清理日志', interactive=False)
        flush = gr.Button('刷新日志')
    with log_group: logger = gr.TextArea(label='Output Logs')
    with gr.Group(visible=False) as eval_group: 
        (evals, eval_input, eval_code, eval_info
        ) = build_eval_group(project, trial, eval_btn, saves)
    record_row = gr.Row(visible=False, equal_height=True)
    with record_row, gr.Column(scale=2):
        rec_md = gr.Code('', language='markdown', show_label=False, 
        show_line_numbers=False, wrap_lines=True)
    with record_row, gr.Column(scale=3):
        view_rec = gr.Markdown(container=True, min_height=800)
    node_row = gr.Row(visible=False, equal_height=True)
    schedule_row = gr.Row(visible=False, equal_height=True)
    file_row = gr.Row(visible=False, equal_height=True)
    monitor_row = gr.Row(visible=False, equal_height=True)
    with monitor_row: notify_users = gr.Textbox(label='Notify Users')
    export_btn.click(on_ckpt_step_change, 
        [project, trial, model], ckpt_step)
    selected = gr.Button('log', visible=False, elem_id='selected')
    code_frame = lambda code: f'''<iframe allowfullscreen
    src='http://{HOST}:8414/?folder={code}' 
    style="width: 100%; height: 90vh" frameborder='0'> </iframe>'''
    code_html = gr.HTML(visible=False, padding=False, autoscroll=True)
    code_btn.click(code_frame, code, code_html)
    record_btn.click(on_record_click, 
        [project, record_btn, rec_md], rec_md)
    rec_md.change(lambda x: x, rec_md, view_rec, show_progress=False)
    tb_frame = lambda p, t: f'''<iframe allowfullscreen src='
    http://{HOST}:8420/?runFilter={p}/{t}#scalars&regexInput={p}/{t}' 
    style="width: 100%; height: 90vh" frameborder='0'> </iframe>'''
    tb_html = gr.HTML(visible=False, padding=False, autoscroll=True)
    tb_btn.click(tb_frame, [project, trial], tb_html)
    operates = [export_btn, eval_btn, record_btn, node_btn, code_btn, 
    schedule_btn, file_btn, monitor_btn, tb_btn, log_btn]
    on, off = gr.update(visible=True), gr.update(visible=False)
    unselect = lambda op: gr.update(variant='secondary', 
        value=op.value, link=None)
    select = lambda op: gr.update(variant='primary', value=
        '保存' if op in [record_btn] else op.value)
    tabs = [ckpt_step, eval_group, record_row, node_row, code_html, 
    schedule_row, file_row, monitor_row, tb_html, log_group]
    get_operate_click_event_args = lambda op: (lambda: [op.elem_id
        ] + [on if op is o else off for o in operates] + [
    select(o) if op is o else unselect(o) for o in operates], 
        None, [selected] + tabs + operates)
    for op in operates: op.click(*get_operate_click_event_args(op))
    PARAMS = { 'TEMPLATE': (template, {}), 
    'DIST_TASK_DEPS': (task_deps, {}),
    'TASKS': (tasks, {}), 'DIST_TASK_TYPE': (task_type, {}), 
    'DIST_DEVICE_KIND': (device_kind, {}), 
    'DIST_NUM_DEVICES': (device_num, {}),
    'DIST_DEVICE_CPUS': (device_cpus, {}),
    'DIST_DEVICE_MEMORY': (device_memory, {}),
    'DIST_NUM_NODES': (node_num, {}),
    'DIST_CPU_KIND': (cpu_kind, set(TASK_CHOICES) - {'RAY'}),
    'DIST_NUM_CPUS': (cpu_num, set(TASK_CHOICES) - {'RAY'}),
    'DIST_CPU_MEMORY': (
        cpu_memory, set(TASK_CHOICES) - {'RAY'}),
    'DIST_NUM_CPU_NODES': (
        cpu_node_num, set(TASK_CHOICES) - {'RAY'}),
    'DIST_NUM_INSTANCES': (
        instance_num, set(TASK_CHOICES) - SERVE_TASKS),
    'DIST_MODEL': (model, {'RAY', 'NONE', 'WEB', 'API', 'SERVE'}),
    'DIST_TRAIN_TYPE': (train_type, NO_TRAIN_TASKS),
    'DIST_LORA_RANK': (lora_rank, NO_TRAIN_TASKS),
    'DIST_LORA_DROPOUT': (lora_dropout, NO_TRAIN_TASKS),
    'DIST_LORA_ALPHA': (lora_alpha, NO_TRAIN_TASKS),
    'DIST_LORA_MODULE': (lora_module, NO_TRAIN_TASKS),
    'DIST_QUANTIZE': (quantize, NO_TRAIN_TASKS), 
    'DIST_BOOST': (boost, NO_TRAIN_TASKS),
    'DIST_EP_SIZE': (ep_size, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_CP_SIZE': (cp_size, NO_TRAIN_TASKS),
    'DIST_TP_SIZE': (tp_size, NO_TRAIN_TASKS - {'EVAL'}), 
    'DIST_PP_SIZE': (pp_size, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_DEEPSPEED_STAGE': (
        deepspeed_stage, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_DATASET_SPLIT': (dataset_split, {}),
    'DATASETS': (datasets, {'RAY', 'NONE', 'WEB', 'API', 'SERVE'}), 
    'DIST_CHAT_TEMPLATE': (
        chat_template, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_SYSTEM_PROMPT': (
        system_prompt, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_MAX_SEQ_LEN': (max_seq_len, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_MAX_GEN_LEN': (max_gen_len, NO_TRAIN_TASKS - {'EVAL'}),
    'DIST_EPOCH': (epoch, NO_TRAIN_TASKS), 
    'DIST_BATCH_SIZE': (batch_size, NO_TRAIN_TASKS),
    'DIST_GRAD_ACCUM': (grad_accum, NO_TRAIN_TASKS), 
    'DIST_LR_RATE': (lr_rate, NO_TRAIN_TASKS),
    'DIST_LR_SCHEDULER': (lr_scheduler, NO_TRAIN_TASKS),
    'DIST_GRAD_CLIP': (grad_clip, NO_TRAIN_TASKS),
    'DIST_VAL_STEP': (val_step, NO_TRAIN_TASKS), 
    'DIST_SAVE_STEP': (save_step, NO_TRAIN_TASKS),
    'DIST_SAVE_LIMIT': (save_limit, NO_TRAIN_TASKS),
    'DIST_LOG_STEP': (log_step, NO_TRAIN_TASKS),
    'DIST_NUM_GENS': (gen_num, set(TASK_CHOICES) - RL_TASKS),
    'DIST_GEN_TEMP': (
        gen_temp, set(TASK_CHOICES) - RL_TASKS),
    'DIST_TOP_P': (top_p, set(TASK_CHOICES) - RL_TASKS),
    'DIST_TOP_K': (top_k, set(TASK_CHOICES) - RL_TASKS),
    'REWARDS': (rewards, set(TASK_CHOICES) - RL_TASKS), 
    'DIST_CONDA_ENV': (conda, {}), 'ENV_CODE': (env_code, {}),
    'DIST_DOCKER_IMAGE': (docker_image, {}), 
    'DIST_USER': (user, {}), 'DIST_USER_GROUP': (user_group, {}),
    'DIST_RESOURCE_GROUP': (resource_group, {}), 
    'DIST_NODE_ALIVE': (node_alive, {}), 
    'DIST_RESOURCE_CONFIG': (resource_cfg, {}),
    'DIST_CODE': (code, {}), 'DIST_SCRIPT': (script, {}),
    'DIST_SCRIPT_ARGS': (script_args, {}),
    'EVALS': (evals, {}), 'EVAL_INPUT': (eval_input, {}),
    'EVAL_CODE': (eval_code, set(TASK_CHOICES)),
    'DIST_NOTIFY_USERS': (notify_users, {}),
    'METRIC': (metric, {}), 'AXIS_X': (axis_x, {}), 
    'METRIC_LABEL': (metric_label, {}), 'NODE': (node, {}),
    'FILTER_LOG': (log_filter, {}), 'DATASET': (dataset, {})}
    train_type.change(on_train_type_select, train_type, lora_row)
    NAMES, VALUES = list(PARAMS), list(v[0] for v in PARAMS.values())
    DEFAULTS = [v[0].value for v in PARAMS.values()]
    tasks.input(update_tasks_type_and_status, tasks, tasks)
    task_deps_change_event_args = (lambda x, y: gr.update(
        visible=x=='' or y=='ON'), [trial, task_deps], tasks)
    task_deps.change(*task_deps_change_event_args)
    task_type.change(on_task_type_change, [task_type] + VALUES, 
        [dataset_group, reward_group] + VALUES)
    project.focus(on_project_input, project, trial)
    project.blur(lambda: gr.update(choices=sorted(TRIALS)), 
        None, project, show_progress='hidden')
    project.input(on_project_input, project, trial
    ).then(lambda t: (None, t or ''), trial, [trial, template]
    ).then(lambda t: t, template, trial)
    on_metric_select_event_args = (on_metric_select, 
        [project, trial, metric, metric_label, axis_x], plot)
    update_task_status_event_args = (update_task_status, 
        [project, trial], 
    [delete, dependent, save, launch, stop, resume, clean])
    gr.Timer(2).tick(*update_task_status_event_args).then(
    update_tasks_type_and_status, tasks, tasks, show_progress=False)
    trial.change(lambda: '', None, output).then(
        on_trial_change, [template, project, trial], VALUES
    ).then(*update_task_status_event_args
    ).then(*task_deps_change_event_args).then(*init_ds_event_args
    ).then(build_config, [project, trial] + VALUES, script_cfg
    ).success(lambda _: None, selected, 
    js='(id) => {document.getElementById(id).click()}')
    trial.blur(on_project_input, project, trial)
    delete.click(delete_trial, [project, trial, delete], 
        [trial, delete, cancel, save, output])
    save.click(save_trial, [project, trial] + VALUES, 
        [project, trial, delete, output])
    on, off = gr.update(visible=True), gr.update(visible=False)
    cancel.click(lambda: ('删除任务', off, on), None, 
        [delete, cancel, save])
    launch.click(save_trial, [project, trial] + VALUES, 
        [project, trial, delete, output]).then(
        launch_trial, [project, trial] + VALUES, output
    ).then(*update_task_status_event_args)
    stop.click(stop_trial, [project, trial], output).then(
        *update_task_status_event_args)
    resume.click(save_trial, [project, trial] + VALUES, 
        [project, trial, delete, output]).then(
    resume_trial, [project, trial] + VALUES, output
    ).then(*update_task_status_event_args)
    for v in VALUES: v.input(verify_trial, [project, trial] + VALUES, 
        VALUES + [output], show_progress='hidden')
    model.blur(lambda: gr.update(visible=True), outputs=algo_coloumn)
    model.focus(lambda: gr.update(visible=False), 
        outputs=algo_coloumn, show_progress='hidden')
    for e in [device_kind.input, device_kind.focus]: 
        e(on_device_kind_change, [project, device_kind], device_num)
    for e in [device_kind.input, device_kind.focus, 
        device_num.input]: e(on_cpu_kind_change, 
        [project, device_kind, device_num], device_cpus)
    for e in [device_kind.blur, project.focus, trial.focus]: e(
        lambda p: gr.update(choices=list(
        get_devices(p, None))), project, device_kind)
    for e in [device_num.input, device_num.focus]: e(
        on_device_num_change, 
    [project, device_kind, device_num, device_cpus], node_num)
    for e in [cpu_kind.input, cpu_kind.focus]: e(
        on_cpu_kind_change, [project, cpu_kind], cpu_num)
    for e in [cpu_kind.blur, project.focus, trial.focus]: e(
        lambda p: gr.update(choices=list(
        get_devices(p, None, True))), project, cpu_kind)
    for e in [cpu_num.input, cpu_num.focus]: e(on_cpu_num_change, 
        [project, cpu_kind, cpu_num], cpu_node_num)
    reward.select(on_reward_select, [reward, rewards], rewards)
    script.focus(build_config, [project, trial] + VALUES, script_cfg)
    ckpt_step.input(on_ckpt_step_change,
        [project, trial, model, ckpt_step], ckpt_step)
    flush_event_args = (flush_log_and_metric, [project, trial, metric, 
        log_filter, node], [metric, metric_label, node, logger])
    for c in [log_btn, flush]: c.click(
        *flush_event_args).then(*on_metric_select_event_args
    ).then(*update_task_status_event_args
    ).then(lambda: '清理日志', None, clean)
    for c in [log_filter, node]: c.input(*flush_event_args)
    for c in [metric, metric_label, axis_x]: 
        c.change(*on_metric_select_event_args)
    metric_label.change(lambda x: gr.update(show_label=not x), 
        metric_label, metric_label)
    clean.click(clean_log, [project, trial, clean, node], 
        [clean, output]).then(
    *flush_event_args).then(*update_task_status_event_args)
    platform.load(initialize_platform, None, [project, trial] + VALUES
    ).then(lambda: (gr.update(choices=MODELS), 
    gr.update(choices=REWARDS), gr.update(choices=CONDAS), 
    gr.update(choices=SCRIPTS)), None, [model, reward, conda, script])

platform = platform.queue(default_concurrency_limit=10)
# platform.launch(server_name='0.0.0.0', server_port=PORT, share=True)
app = gr.mount_gradio_app(app, platform, path='/aigc/train')
app.routes.insert(0, app.routes.pop())
uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')