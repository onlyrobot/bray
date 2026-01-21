import logging, asyncio, time, os, fastapi, uvicorn
import subprocess, signal, random
from bray.common import (HOST,
    save_task_config, load_task_config, load_trial_config,
    get_trial_path, get_project_path, save_trial_config)
from bray.common import cached_session, request
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
START_TIME, START_WAIT = time.time(), 10
MASTER_PORTS, PORT = [str(8614 + i) for i in range(10)], 8413
class NodeInfo:
    def __init__(self, env, kind, gpu, cpu):
        self.reset(env, kind, gpu, cpu)
        self.cond, self.task = asyncio.Condition(), None
        self.tasks: 'list[dict]' = []; self.mtime = time.time()
        self.used_ports, self.routers = [], None
        self.used_gpus, self.used_cpus = [], []
    def reset(self, env, kind, gpu, cpu):
        self.env = {} if env is None else env
        self.kind, self.gpu, self.cpu = kind, gpu, cpu
    def is_share(self) -> bool:
        return not self.env.get('DIST_NODE_ALIVE')
    def is_alive(self) -> bool: return self.kind and (
        self.is_share() or self.used_ports or self.used_cpus or int(
        self.env['DIST_NODE_ALIVE']) + self.mtime > time.time())
    def gpus(self, remain=True) -> list: 
        return [i for i in range(self.gpu or 0) if 
        not remain or i not in self.used_gpus]
    def free(self, gpus, cpus):
        self.used_gpus = list(set(self.used_gpus) - set(gpus))
        self.used_cpus = list(set(self.used_cpus) - set(cpus))
        self.mtime = time.time()
    def cpus(self, remain=True) -> list: 
        return [i for i in range(self.cpu or 0) 
        if not remain or i not in self.used_cpus]
    def port(self, port=None) -> int:
        port = port or [p for p in MASTER_PORTS if p not 
        in self.used_ports].pop(0)
        self.used_ports.append(port); return port
    async def add_task(self, env: dict):
        self.tasks.append(env)
        async with self.cond: self.cond.notify_all()
HOST2NODE_INFO: 'dict[str: NodeInfo]' = {}
class TaskInfo:
    def __init__(self, env: dict, deps: set):
        self.cond, self.task = asyncio.Condition(), None
        self.env, self.deps, self.msg = env, deps, ''
        self.resources: 'dict[tuple, dict[str: tuple[list]]]' = {}
    def type(self): return self.env.get('DIST_TASK_TYPE')
    def status(self) -> str: 
        return self.env.get('DIST_TASK_STATUS') or 'UNKNOWN'
    def update_status(self, status: str): 
        self.env['DIST_TASK_STATUS'] = status
    def is_serve(self): return 'DIST_NUM_INSTANCES' in self.env
    def is_done(self) -> bool: 
        return self.status() in ['SUCCESS', 'FAILED', 'STOPPED']
    def is_ready(self) -> bool:
        if self.is_serve(): return bool(self.resources)
        return self.status() == 'SUCCESS'
    def save_env(self, task_id, env=None):
        if env is not None: self.env = env
        try: save_task_config(*task_id.split('/', 1), self.env)
        except: logging.warning(f'fail to save task env')
CREATED_TASK_ID2INFO: 'dict[str: TaskInfo]' = {}
ROUTERS: 'dict[str: list[str]]' = {}
PENDING_TASK_ID2INFO: 'dict[str: TaskInfo]' = {}

CLUSTER = os.environ.pop('DIST_CLUSTER', f'{HOST}:{PORT}')
def set_cluster(c): global CLUSTER; CLUSTER = c; return c

async def restore_task_on_start(wait_for_start: float):
    envs = {f'{p}/{t}': env for p, ts in (await dist_task_query()
        ).items() for t, env in ts.items() if env}
    for task_id, env in envs.items():
        CREATED_TASK_ID2INFO[task_id] = TaskInfo(env, set())
    await asyncio.sleep(wait_for_start)
    await asyncio.gather(*[dist_task_launch(e.copy(), '') 
        for e in envs.values() if e and not 
    (e.get('DIST_DEPENDENT') or e.get('DIST_TASK_STATUS'))])

async def register_to_master_on_start(_: fastapi.FastAPI):
    asyncio.create_task(restore_task_on_start(START_WAIT + 1))
    env = {k: v for k, v in os.environ.items() if 
        k.startswith('DIST_')}
    if CLUSTER == f'{HOST}:{PORT}': device_kind = 'localhost'
    else: device_kind = env.pop('DIST_DEVICE_KIND', '')
    t = asyncio.create_task(register_dist_node(env, device_kind))
    t.add_done_callback(lambda _: handle_exit_signal()); yield

app = fastapi.FastAPI(lifespan=register_to_master_on_start)

async def notify_all_nodes(nodes: 'list[NodeInfo]'):
    async with nodes[0].cond: nodes[0].cond.notify_all()
    if not (nodes := nodes[1:]): return
    asyncio.create_task(notify_all_nodes(nodes))

def insert_router(task_id: str, host: str, port: str):
    if task_id not in ROUTERS: ROUTERS[task_id] = []
    if (rule := f'{host}:{port}') in ROUTERS[task_id]: return
    (rules := ROUTERS[task_id]).append(rule)
    for i in (nodes := list(HOST2NODE_INFO.values())): 
        if i.routers is not None: i.routers[task_id] = rules
    asyncio.create_task(notify_all_nodes(nodes))

def remove_router(task_id: str, host: str, port: str):
    if task_id not in ROUTERS: return
    if (rule := f'{host}:{port}') not in ROUTERS[task_id]: return
    (rules := ROUTERS[task_id]).remove(rule)
    for i in (nodes := list(HOST2NODE_INFO.values())): 
        if i.routers is not None: i.routers[task_id] = rules
    asyncio.create_task(notify_all_nodes(nodes))
    if not ROUTERS[task_id]: del ROUTERS[task_id]

async def launch_task_dep(task: list, dep: str, env: dict) -> str:
    if len(parts := (task_id := task[0]).split('/', 1)) != 2: 
        return f'实验不存在 {task[0]}' if task[0] else ''
    if isinstance(env := env.get(task[1], {}), str): env = {}
    if not env and (info := CREATED_TASK_ID2INFO.get(task_id)):
        if info.is_ready(): return ''
    return await dist_task_launch(env, *parts, dep, task[-1])

def clone_trial(template: str, trial_path: str) -> str:
    if not template or len(parts := template.split('/', 1)) != 2: 
        return f'实验模版不存在 {template}'
    template_path = get_trial_path(*parts)
    if not os.path.exists(f'{template_path}/config.json'): 
        return f'实验模版不存在 {template}'
    os.makedirs(trial_path, exist_ok=True)
    os.system(f'cp {template_path}/*.* {trial_path}')

def update_dep_task(env: dict, task: list):
    task_id, task_name, template = task[0], task[1], task[-1]
    if (config := env.get(task_name)) is None: return
    if isinstance(config, str): task[0] = config; return
    if len(parts := task_id.split('/', 1)) != 2: parts = ['', '']
    project = config.get('DIST_PROJECT') or parts[0]
    task[0] = f'{project}/{config.get("DIST_TRIAL") or parts[1]}'
    if t := config.get('DIST_TEMPLATE'): task[-1] = t

def update_trial_by_env(config: dict, env: dict):
    for d in config.get('DATASETS', []): d[2] = env.get(d[0], d[2])
    datasets = [d for d in config.get('DATASETS', []) if d[2]]
    if dataset := ' '.join([d[2] for d in datasets if d[0]]):
        env['DIST_DATASET'] = dataset
    kinds = set(k for k in [d[0].split('_DATASET')[0] for 
        d in datasets if d[0]] if k != 'ANY')
    for k in kinds: env[f'{k}_DATASET'] = ' '.join([d[2] for 
        d in datasets if d[0].startswith(k)] )

    for t in config.get('TASKS', []): update_dep_task(env, t)
    tasks = [t for t in config.get('TASKS', []) if t[0]]
    if not tasks or config['DIST_TASK_DEPS'] != 'ON': return
    config['DIST_TASKS'] = ' '.join([t[0] for t in tasks])

def get_latest_ckpt_path(project: str, trial: str) -> str:
    path = f'{get_trial_path(project, trial)}/output'
    handle_ckpt = lambda x: int(x.removeprefix('checkpoint-'))
    is_ckpt = lambda x: x.startswith('checkpoint-')
    ckpts = [] if not os.path.exists(path) else sorted([handle_ckpt(x) 
        for x in os.listdir(path) if is_ckpt(x)])
    return f'{path}/checkpoint-{ckpts[-1]}' if ckpts else ''

@app.post('/dist/task/launch')
async def dist_task_launch(env={}, project='', trial='', dep='', 
        template='', resume: bool=None, 
        request: fastapi.Request=None) -> str:
    if time.time() - START_TIME < START_WAIT: return f'请稍后重试...'
    if not env and request and await request.body(): 
        env = await request.json()
    if not project: project = env.pop('DIST_PROJECT', '')
    if not trial: trial = env.pop('DIST_TRIAL', '')
    if not template: template = env.get('DIST_TEMPLATE', '')
    if len(parts := template.split('/', 1)) == 2:
        project, trial = project or parts[0], trial or parts[1]
    if not project or not trial: return '缺失实验名'
    trial_path = get_trial_path(project, trial)
    if not os.path.exists(f'{trial_path}/config.json'): 
        if r := clone_trial(template, trial_path): return r
    config = load_trial_config(project, trial)
    if env: update_trial_by_env(config, env)
    is_task_serve = config['DIST_TASK_TYPE'] == 'SERVE'
    task_exist = os.path.exists(f'{trial_path}/output/out.0.txt')
    resumable = is_task_serve or config.get('DIST_RESUMABLE')
    if task_exist and resume is None: resume = True
    if resume and task_exist and not resumable:
        return '任务不支持断点续跑，请重新清理日志后重试'
    if not resume and task_exist: 
       return '当前任务存在日志和检查点，请清理后再启动'
    dist_device_kind = config.get('DIST_DEVICE_KIND')
    dist_cpu_kind = config.get('DIST_CPU_KIND')
    if not (dist_cpu_kind or dist_device_kind):
        return '未选择设备类型或CPU类型'
    if config.get('DIST_NUM_NODES') and not dist_device_kind:
        return '填写了设备节点数但未填写设备类型'
    num_devices = config.get('DIST_NUM_DEVICES')
    device_cpus = config.get('DIST_DEVICE_CPUS')
    if dist_device_kind and not (num_devices or device_cpus):
        return f'未选择设备{dist_device_kind}的数量'
    if config.get('DIST_NUM_CPU_NODES') and not dist_cpu_kind:
        return '填写了CPU节点数但未填写CPU类型'
    if dist_cpu_kind and not config.get('DIST_NUM_CPUS'):
        return f'未选择{dist_cpu_kind}的CPU数量'
    config |= {k: type(config.get(k, v))(v) for k, v in env.items() 
        if not isinstance(v, (list, dict))}
    e = {k: str(v) for k, v in config.items() if not 
        isinstance(v, (list, dict))}
    e |= {'DIST_PROJECT': project, 'DIST_TRIAL': trial,
    'DIST_TRIAL_PATH': trial_path, 'DIST_DEPENDENT': dep or ''}
    if resume: e['DIST_RESUME_PATH'
        ] = get_latest_ckpt_path(project, trial) or ''
    if r := await dist_task_create(e, dep): return f'{r} {dep} '
    if env: save_trial_config(project, trial, config)
    if config['DIST_TASK_DEPS'] != 'ON': 
        return '' if dep else f'任务启动成功'
    dep_tasks = [launch_task_dep(t, f'{project}/{trial}', env) 
        for t in config.get('TASKS', [])]
    if not any(rs := await asyncio.gather(*dep_tasks)): 
        return '' if dep else '任务创建成功，等待依赖就绪...'
    stop_msg = '' if dep else await dist_task_stop(project, trial)
    return f'{"".join(rs)} 在创建子任务时失败 {stop_msg}'

@app.post('/dist/task/create')
async def dist_task_create(env: dict, dep: str='') -> str:
    task_id = f'{env["DIST_PROJECT"]}/{env["DIST_TRIAL"]}'
    logging.info(f'start to create task {env}')
    if not (task := CREATED_TASK_ID2INFO.get(task_id)):
        task = CREATED_TASK_ID2INFO[task_id] = TaskInfo({}, set())
    elif not dep and dep in task.deps: return '重复启动'
    exist = task.resources or task_id in PENDING_TASK_ID2INFO
    if (exist or task.is_ready()) and dep == '':
        return '任务已经就绪，无需重新启动'
    ignores = ['DIST_RESUME_PATH', 'DIST_DEPENDENT']
    if (exist or task.is_ready()) and any(v != task.env.get(k) for 
        k, v in env.items() if k not in ignores):
        return '任务已经就绪，且两次启动的参数不一致，请修改后重试'
    if dep is not None: task.deps.add(dep)
    if not task.env or not exist and not task.is_ready():
        task.save_env(task_id, env)
    if dep == '' and dep in task.deps: 
        await notify_user(task_id, 'CREATED', '创建成功')
    if not (r := await schedule_task(task_id, task)): return ''
    task_deps_on = task.env['DIST_TASK_DEPS'] == 'ON'
    if task_deps_on or '' not in task.deps or dep: return ''
    await remove_task_and_clean(task_id, 'FAILED'); return r
    # if dep in task.deps: task.deps.remove(dep)
    # task.update_status('FAILED'); task.save_env(task_id)
    # await notify_user(task_id, 'FAILED', '调度失败')
    # PENDING_TASK_ID2INFO.pop(task_id, None); return r

def is_task_deps_ready(task_id: str, root=True) -> bool:
    if not (task := CREATED_TASK_ID2INFO.get(task_id)): return False
    if task.is_ready() and task.is_done(): return True
    is_self_ready = True if root else task.is_ready()
    return is_self_ready and all(is_task_deps_ready(t, root=False
    ) for t in task.env.get('DIST_TASKS', '').split(' ') if t)

def allocate_cpu_total(host2infos, cpu, share=True) -> list:
    allocated, index, remain = [], -1, cpu
    while remain > 0 and (index := index + 1) < len(host2infos):
        a = allocate_cpu_total_(host2infos[index], remain, share)
        if a: allocated.append(a); remain -= len(a[2])
    return allocated if remain == 0 else None

def allocate_cpu_total_(h2i, remain, share=True) -> tuple:
    s = share and h2i[1].is_share()
    if not s and len(h2i[1].cpus()) > remain: return None
    return h2i, [], h2i[1].cpus()[:remain]

def allocate_cpu_single(host2infos, num, share=True) -> tuple:
    valid = lambda x, s: x >= num if share and s else x == (num or x)
    for index, h2i in enumerate(host2infos):
        if valid(len(h2i[1].cpus()), h2i[1].is_share()): break
    else: return None   # not any valid cpu
    host2infos.pop(index); return h2i, [], h2i[1].cpus()[:num]

def allocate_cpu(host2infos, num_cpus, share=True) -> list:
    allocated, host2infos = [], host2infos.copy()
    for n in reversed(num_cpus):
        allocated.append(allocate_cpu_single(host2infos, n[1], share))
        if not allocated[-1]: return None
    sorted_allocated = [None] * len(allocated)
    for i, n in enumerate(reversed(num_cpus)): 
        sorted_allocated[n[0]] = allocated[i]
    return sorted_allocated if all(allocated) else None

def allocate_gpu_total(host2infos, gpu, cpu, share=True) -> list:
    if not gpu: return allocate_cpu_total(host2infos, cpu or 0, share)
    allocated, index, remain = [], -1, gpu
    while remain > 0 and (index := index + 1) < len(host2infos):
        a = allocate_gpu_total_(host2infos[index], remain, cpu, share)
        if a: remain -= len(a[1]); allocated.append(a)
    return allocated if remain == 0 else None

def allocate_gpu_total_(h2i, remain, cpu, share=True) -> tuple:
    if not (info := h2i[1]).gpus(): return None
    if cpu is None: cpu = len(info.cpus()) // len(info.gpus())
    max_num = len(info.cpus()) // max(1, cpu)
    gpu_num = min(remain, max_num, len(info.gpus()))
    if not gpu_num or not (cpu_num := cpu * gpu_num): return None
    s = share and info.is_share()
    if not s and gpu_num != len(info.gpus()): return None
    if not s and cpu_num != len(info.cpus()): return None
    return h2i, info.gpus()[:gpu_num], info.cpus()[:cpu_num]

def allocate_gpu_single(host2infos, gpu, cpu, share=True) -> tuple:
    valid = lambda x, y, s: x >= y if share and s else x == (y or x)
    for index, h2i in enumerate(host2infos):
        gpus, cpus = h2i[1].gpus(), h2i[1].cpus()
        if valid(len(gpus), gpu, s := h2i[1].is_share()) and valid(
        len(cpus), cpu or len(cpus), s): break
    else: return None   # not any valid gpu
    if cpu is None: cpu = gpu * len(cpus) // len(gpus)
    host2infos.pop(index); return h2i, gpus[:gpu], cpus[:cpu]

def allocate_gpu(host2infos, num_gpus_cpus, share=True) -> list:
    allocated, host2infos = [], host2infos.copy()
    for n in reversed(num_gpus_cpus):
        allocated.append(allocate_gpu_single(host2infos, *n[1], share))
        if not allocated[-1]: return None
    sorted_allocated = [None] * len(allocated)
    for i, n in enumerate(reversed(num_gpus_cpus)): 
        sorted_allocated[n[0]] = allocated[i]
    return sorted_allocated if all(allocated) else None

def allocate_gpu_if_needed(host2infos, env, share=True) -> list:
    num_nodes = int(env.get('DIST_NUM_NODES') or 0)
    num_devices = [int(env.get('DIST_NUM_DEVICES') or 0) for _ in 
        range(num_nodes)] + [
    int(env[n]) for n in env if n.startswith('DIST_NUM_DEVICES_')]
    device_cpus = env.get('DIST_DEVICE_CPUS') or None
    if device_cpus: device_cpus = int(device_cpus)
    num_cpus = [device_cpus for _ in range(num_nodes)] + [
        int(env[n]) for n in env 
    if n.startswith('DIST_DEVICE_CPUS_')]
    while len(num_cpus) < len(num_devices): num_cpus.append(None)
    num_gpus_cpus = enumerate(zip(num_devices, num_cpus))
    num_gpus_cpus = sorted(num_gpus_cpus, key=lambda x: x[1][0])
    return allocate_gpu(host2infos, num_gpus_cpus, share)

def allocate_cpu_if_needed(host2infos, env, share=True) -> list:
    num_cpus = [int(env.get('DIST_NUM_CPUS') or 0) for _ in range(
        int(env.get('DIST_NUM_CPU_NODES') or 0))] + [
    int(env[n]) for n in env if n.startswith('DIST_NUM_CPUS_')]
    num_cpus = sorted(enumerate(num_cpus), key=lambda x: x[1])
    return allocate_cpu(host2infos, num_cpus, share)

def merge_a_to_allocated(allocated: list, a: tuple):
    a[0][1].used_gpus += a[1]; a[0][1].used_cpus += a[2]
    allocated_hosts = [x[0][0] for x in allocated]
    try: target_a = allocated[allocated_hosts.index(a[0][0])]
    except ValueError: allocated.append(a); return
    target_a[1].extend(a[1]); target_a[2].extend(a[2])

async def schedule_all_tasks(task_ids: tuple=None, index=0):
    if task_ids is None: task_ids = tuple(PENDING_TASK_ID2INFO)
    if not task_ids or index >= len(task_ids): return
    task_id = task_ids[index]; index += 1
    await schedule_task(task_id, CREATED_TASK_ID2INFO[task_id])
    asyncio.create_task(schedule_all_tasks(task_ids, index))

async def notify_user(dep: str, status: str, msg, task_id=''):
    if not (task := CREATED_TASK_ID2INFO.get(dep)): return
    if not task_id: task_id = dep
    for d in task.deps - {''}: await notify_user(
        d, status, f'的子任务\n{dep}\n{msg}', task_id)
    if '' not in task.deps: return
    if not (user := task.env.get('DIST_NOTIFY_USERS')): return
    from bray.monitor import try_notify_user_msg
    await try_notify_user_msg(user, dep, msg, status, task_id)

async def schedule_task(task_id, task: TaskInfo, retry=0) -> str:
    if retry > 0: await asyncio.sleep(retry * 60)
    need_allocate_resource = task_id not in PENDING_TASK_ID2INFO
    r = await schedule_task_(task_id, task)
    if r != '没有足够的GPU/CPU/PORT资源，请准备后重试...': return r
    if not need_allocate_resource: return r
    if not (r := await allocate_resource(task_id, task)): return ''
    PENDING_TASK_ID2INFO.pop(task_id, None)
    asyncio.create_task(schedule_task(task_id, task, retry + 1))
    logging.warning(f'allocate resource err {r}'); return r

async def allocate_resource(task_id, task: TaskInfo):
    try: r = await allocate_resource_(task, f'{HOST}:{PORT}')
    except Exception as e: r = f'allocate resource err {e}'
    if r: await notify_user(task_id, 'ERROR', r); return r

async def allocate_resource_(task: TaskInfo, cluster: str) -> str:
    env = {**os.environ, **task.env, 'DIST_CLUSTER': cluster}
    process = await asyncio.create_subprocess_shell(
        'source ./allocate.sh', env=env, shell=True,
    start_new_session=True, stderr=asyncio.subprocess.PIPE)
    return (await process.communicate())[1].decode()

async def schedule_task_(task_id, task: TaskInfo) -> str:
    if task.resources: return '任务正在运行中'
    if not task.env or not task.deps: return '任务已经结束'
    if task.is_ready(): return '任务已经就绪'
    if not is_task_deps_ready(task_id): return '依赖任务未就绪'
    PENDING_TASK_ID2INFO[task_id] = task; env = task.env
    node_affinity = env.get('DIST_NODE_AFFINITY', '')
    host2infos = [] if not env.get('DIST_DEVICE_KIND') else [
        (h, i) for h, i in HOST2NODE_INFO.items()
        if i.kind == env['DIST_DEVICE_KIND']
    and all(v == env.get(k, v) for k, v in i.env.items())]
    def remain_gpu_num(h2i: tuple) -> float:
        priority = 0.0 if h2i[0] in node_affinity else 10000
        return len(h2i[1].gpus()) + priority
    host2infos = sorted(host2infos, key=remain_gpu_num)

    def free_used_gpu_and_cpu(allocated: list) -> str:
        for a in allocated: a[0][1].free(a[1], a[2])
        return '没有足够的GPU/CPU/PORT资源，请准备后重试...'
    
    share = not env.get('DIST_NODE_EXCLUSIVE')
    allocated = allocate_gpu_if_needed(host2infos, env, share)
    if allocated is None: return free_used_gpu_and_cpu([])
    for a in allocated: a[0][1].used_gpus += a[1]
    for a in allocated: a[0][1].used_cpus += a[2]
    host2infos = sorted(host2infos, key=remain_gpu_num)

    num_devices = int(env.get('DIST_NUM_DEVICES') or 0)
    device_cpus = env.get('DIST_DEVICE_CPUS') or None
    if device_cpus: device_cpus = int(device_cpus)
    if env.get('DIST_NUM_NODES'): num_devices = device_cpus = 0
    extra = allocate_gpu_total(
        host2infos, num_devices, device_cpus, share)
    if extra is None: return free_used_gpu_and_cpu(allocated)
    for a in extra: merge_a_to_allocated(allocated, a)

    host2infos = [] if not env.get('DIST_CPU_KIND') else [
        (h, i) for h, i in HOST2NODE_INFO.items() 
        if  i.kind == env['DIST_CPU_KIND'] 
    and all(v == env.get(k, v) for k, v in i.env.items())]
    def remain_cpu_num(h2i: tuple) -> float:
        priority = 0.0 if h2i[0] in node_affinity else 10000
        return len(h2i[1].cpus()) + priority
    host2infos = sorted(host2infos, key=remain_cpu_num)

    extra = allocate_cpu_if_needed(host2infos, env, share)
    if extra is None: return free_used_gpu_and_cpu(allocated)
    for a in extra: merge_a_to_allocated(allocated, a)
    host2infos = sorted(host2infos, key=remain_cpu_num)

    num_cpus = int(env.get('DIST_NUM_CPUS') or 0)
    if env.get('DIST_NUM_CPU_NODES'): num_cpus = 0
    extra = allocate_cpu_total(host2infos, num_cpus, share)
    if extra is None: return free_used_gpu_and_cpu(allocated)
    for a in extra: merge_a_to_allocated(allocated, a)

    env = env | {'DIST_NUM_NODES': str(len(allocated))}
    master = env['DIST_MASTER'] = allocated[0][0][0]
    port = env['DIST_MASTER_PORT'] = HOST2NODE_INFO[master
        ].port(env.get('DIST_MASTER_PORT'))
    if task.is_serve(): insert_router(task_id, master, port)
    task.resources[(master, port)] = {
        a[0][0]: (a[1], a[2]) for a in allocated}
    async def change_status_later(timeout=30):
        await asyncio.sleep(timeout)
        await remove_task_and_clean(task_id, 'FAILED')
    PENDING_TASK_ID2INFO.pop(task_id, None)
    task.task = asyncio.create_task(change_status_later())
    await asyncio.gather(*[node.add_task(env | {
        'DIST_WORKER': host, 'DIST_NODE_RANK': str(i),
        'DIST_NUM_CPUS': str(len(cpus)), 
        'DIST_CPUS': ','.join(map(str, cpus))} | (
    {} if not gpus else {'DIST_NUM_DEVICES': str(len(gpus)),
        'DIST_DEVICES': ','.join(map(str, gpus))}))
    for i, ((host, node), gpus, cpus) in enumerate(allocated)])
    await notify_user(task_id, 'RUNNING', '调度成功')
    for dep in task.deps - {''}: 
        await schedule_task(dep, CREATED_TASK_ID2INFO[dep])
    logging.info(f'scheduled task {env}'); return ''

@app.api_route('/dist/cluster', methods=['POST', 'GET'])
async def dist_cluster(cluster: str='') -> str:
    return CLUSTER if not cluster else set_cluster(cluster)

async def remove_task_and_clean(task_id, status) -> TaskInfo:
    info = CREATED_TASK_ID2INFO.get(task_id)
    if info is None or info.is_done(): return info
    PENDING_TASK_ID2INFO.pop(task_id, None)
    await notify_user(task_id, status, '任务结束')
    info.update_status(status); info.save_env(task_id)
    if '' in info.deps: info.deps.remove('')
    for t in info.env.get('DIST_TASKS', '').split(' '):
        if t: await remove_task_dep(t, task_id)
    if not info.resources: return info
    for m, p in info.resources.keys():
        if info.is_serve(): remove_router(task_id, m, p)
        HOST2NODE_INFO[m].used_ports.remove(p)
    for gcs in info.resources.values():
        for h, gc in gcs.items(): HOST2NODE_INFO[h].free(*gc)
    asyncio.create_task(schedule_all_tasks())
    if info.task: info.task.cancel(); info.task = None
    info.resources.clear(); return info

async def remove_task_dep(task_id: str, dep: str):
    if not (info := CREATED_TASK_ID2INFO.get(task_id)): return
    if dep in info.deps: info.deps.remove(dep)
    if not info.deps: await dist_task_stop(*task_id.split('/', 1))

@app.post('/dist/task/stop')
async def dist_task_stop(project: str, trial: str) -> str:
    info = CREATED_TASK_ID2INFO.get(f'{project}/{trial}')
    if not info or info.is_done(): return ''
    if deps := info.deps - {''}: 
        return f'当前任务被{deps}依赖，无法停止'
    await remove_task_and_clean(f'{project}/{trial}', 'STOPPED')
    async with info.cond: info.cond.notify_all()
    logging.info(f'stopped task {info.env}'); return ''

@app.post('/dist/task/remove')
async def dist_task_remove(project: str, trial: str) -> str:
    info = CREATED_TASK_ID2INFO.get(t := f'{project}/{trial}')
    if info and info.resources: return '任务运行中，无法移除'
    if t in PENDING_TASK_ID2INFO: return '任务等待调度中，无法移除'
    info.env.clear() if info else None; return ''

def dist_task_query_(project: str, trial='') -> dict:
    if info := CREATED_TASK_ID2INFO.get(f'{project}/{trial}'): 
        return {trial: info.env}
    project_path = get_project_path(project)
    if not os.path.isdir(project_path): return {}
    trials = [trial] if trial else os.listdir(project_path)
    trials = [t for t in trials if os.path.isdir(
        get_trial_path(project, t))]
    tasks = {id: i.env for id, i in CREATED_TASK_ID2INFO.items()
        if i and id.startswith(project)}
    def try_load_task(project, trial) -> dict:
        try: return load_task_config(project, trial)
        except: return {}
    tasks = {k: tasks[k] for k in trials if k in tasks}
    tasks.update({id: try_load_task(*id.split('/', 1)) for id in 
    (f'{project}/{t}' for t in trials) if id not in tasks})
    return {k.split('/', 1)[-1]: v for k, v in tasks.items()}

@app.get('/dist/task/query')
async def dist_task_query(project='', trial='') -> dict:
    ps = [project] if project else os.listdir(get_project_path())
    return {p: dist_task_query_(p, trial) 
    for p in ps if os.path.isdir(get_project_path(p))}

@app.get('/dist/task/status')
async def dist_task_status(project: str, trial: str) -> tuple:
    info = CREATED_TASK_ID2INFO.get(t := f'{project}/{trial}')
    if not info: return 'UNKNOWN', ''
    deps = info.deps - {''}; dep = deps.pop() if deps else ''
    if info.resources: return 'RUNNING', dep
    # if t in PENDING_TASK_ID2INFO: return 'PENDING', dep
    return 'PENDING' if info.deps else info.status(), dep

@app.get('/dist/device/query')
async def dist_gpu_query(project: str, remain=True) -> dict:
    host2node_info = {h: i for h, i in HOST2NODE_INFO.items() if 
        i.env.get('DIST_PROJECT', project) == project}
    devices = {i.kind: {} for i in host2node_info.values()}
    for h, i in host2node_info.items():
        devices[i.kind][h] = (i.gpus(remain), i.cpus(remain))
    return {k: d for k, d in devices.items() if d and k}

def add_task_dep(task_id: str, dep: str):
    if t := CREATED_TASK_ID2INFO.get(task_id): t.deps.add(dep)
    else: CREATED_TASK_ID2INFO[task_id] = TaskInfo({}, {dep})

def restore_task(dep: str, env: dict, info: TaskInfo):
    for t in env.get('DIST_TASKS', '').split(' '): add_task_dep(t, dep)
    if not env.get('DIST_DEPENDENT'): info.deps.add('')
    
async def initialize_task_if_needed(env: dict) -> tuple:
    task_id = f'{env["DIST_PROJECT"]}/{env["DIST_TRIAL"]}'
    if not (info := CREATED_TASK_ID2INFO.get(task_id)):
        info = CREATED_TASK_ID2INFO[task_id] = TaskInfo({}, set())
    if info.is_done(): return task_id, info
    if not info.resources: restore_task(task_id, env, info)
    master, host = env['DIST_MASTER'], env['DIST_WORKER']
    master_port = env['DIST_MASTER_PORT']
    gpus_cpus = info.resources.get((master, master_port), {})
    if host in gpus_cpus: return task_id, info
    if not (node := HOST2NODE_INFO.get(host)):
        node = HOST2NODE_INFO[host] = NodeInfo(*([None]*4))
    if not gpus_cpus and info.is_serve(): 
        insert_router(task_id, master, master_port)
    if host == master: node.used_ports.append(master_port)
    gpus = [int(i) for i in env.get(
        'DIST_DEVICES', '').split(',') if i]
    cpus = [int(i) for i in env['DIST_CPUS'].split(',') if i]
    node.used_gpus += gpus; node.used_cpus += cpus
    gpus_cpus[host] = (gpus, cpus)
    info.resources[(master, master_port)] = gpus_cpus
    return task_id, CREATED_TASK_ID2INFO[task_id]

async def wait_timeout(cond: asyncio.Condition, timeout=60):
    async with cond: await asyncio.wait_for(cond.wait(), timeout)

@app.post('/dist/task/register')
async def dist_task_register(env: dict) -> bool:
    task_id, info = await initialize_task_if_needed(env)
    if info.is_done(): return False
    if info.task: info.task.cancel(); info.task = None
    timeout = int(env.get('DIST_REGISTER_TIMEOUT', 60))
    try: await wait_timeout(info.cond, timeout)
    except asyncio.TimeoutError: pass
    if info.task: info.task.cancel(); info.task = None
    if info.is_done(): return False
    status = env.get('DIST_TASK_STATUS', 'FAILED')
    if info.is_serve(): status = 'FAILED'
    async def change_status_later(timeout: float):
        await asyncio.sleep(timeout)
        await remove_task_and_clean(task_id, status)
    info.task = asyncio.create_task(change_status_later(timeout))
    return task_id not in PENDING_TASK_ID2INFO

@app.post('/dist/node/register')
async def dist_node_register(req: fastapi.Request) -> tuple:
    host, env, kgc, routers, task = data = await req.json()
    if not host: data[0] = host = req.client.host
    if host in HOST2NODE_INFO: info = HOST2NODE_INFO[host]
    else: info = HOST2NODE_INFO[host] = NodeInfo(env, *kgc)
    if env != info.env or kgc != [info.kind, info.gpu, 
        info.cpu]: info.reset(env, *kgc)
    if info.task: info.task.cancel(); info.task = None
    if info.tasks and task == info.tasks[0]: info.tasks.pop(0)
    if None in [info.routers, routers]: 
        asyncio.create_task(schedule_all_tasks())
    if routers == info.routers: info.routers = {}
    if routers is None: info.router = None
    try: await wait_for_task(info, timeout=60)
    except asyncio.TimeoutError: pass
    if info.task: info.task.cancel(); info.task = None
    async def change_status_later():
        await asyncio.sleep(60); info.kind = None
    info.task = asyncio.create_task(change_status_later())
    if info.routers is not None: data[-2] = info.routers
    else: data[-2] = ROUTERS; info.routers = {}
    if not info.is_alive(): info.kind = data[2] = None
    if not info.tasks: data[-1] = {}; return data
    logging.info(f'launch task {host} {info.tasks[0]}')
    data[-1] = info.tasks[0]; return data

async def wait_for_task(info: NodeInfo, timeout=60):
    async with info.cond: await asyncio.wait_for(
    info.cond.wait_for(lambda: info.tasks 
    or not info.is_alive() or info.routers != {}), timeout)
    
@app.post('/dist/node/update')
async def dist_node_update(host: str, project: str) -> bool:
    if not (info := HOST2NODE_INFO.get(host)): return True
    if info.used_gpus: return False
    info.env['DIST_PROJECT'] = project; return True

@app.post('/dist/node/remove')
async def dist_node_remove(host: str) -> bool:
    if not (info := HOST2NODE_INFO.get(host)): return True
    if info.used_gpus or info.used_cpus: return False
    HOST2NODE_INFO.pop(host); return True

async def handle(method, url, data, headers, params):
    r = await cached_session().request(method, url, data=data, 
        headers=headers, params=params)
    async def stream_response() -> 'AsyncGenerator':
        async with r:
            async for c in r.content: yield c
    return fastapi.responses.StreamingResponse(
    stream_response(), status_code=r.status, headers=r.headers)

@app.api_route('/{path:path}', methods=['GET', 'POST', 
    'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
async def router(request: fastapi.Request) -> fastapi.Response:
    path = request.url.path; parts = path.split('/')
    if len(parts) > 2: task_id = '/'.join(parts[1:3])
    else: raise fastapi.HTTPException(404, detail='path err')
    if not (rules := ROUTERS.get(task_id)): 
        raise fastapi.HTTPException(404, detail=f'no router')
    target = random.choice(rules)
    url = f'http://{target}/{"/".join(parts[3:])}'
    logging.info(f'router {request.url} to {url}')
    method, body = request.method, await request.body()
    try: return await handle(method, url, 
        body, request.headers, request.query_params)
    except Exception as e: err = e
    logging.warning(f'router {task_id} {request.url} err: {err}')
    raise fastapi.HTTPException(500, detail=f'err: {err}')

async def launch_dist_task(host: str, env: str):
    nnode, node = env['DIST_NUM_NODES'], env['DIST_NODE_RANK']
    nproc_per_node = len(env.get('DIST_DEVICES', '').split(','))
    code = env.get('DIST_CODE') or os.getcwd()
    conda = os.path.join(os.getcwd(), 'conda.sh')
    envs = (f'source {conda} && cd {code} && '
    f'MASTER_ADDR={env["DIST_MASTER"]} NODE_RANK={node} '
    f'MASTER_PORT={env["DIST_MASTER_PORT"]} '
    f'CUDA_VISIBLE_DEVICES={env.get("DIST_DEVICES", "")} '
    f'NPROC_PER_NODE={nproc_per_node} NNODES={nnode} ')
    if script_envs := env.get('DIST_SCRIPT_ENVS'): 
        envs = f'{envs} {script_envs}'
    script = os.path.join(code, s := env['DIST_SCRIPT'])
    if s.startswith('script/') and not os.path.exists(script):
        script = os.path.join(os.getcwd(), s)
    if not os.path.exists(script): script = s
    elif script.endswith('.sh'): script = f'source {script}'
    elif script.endswith('.py'): script = f'python {script}'
    if script_args := env.get('DIST_SCRIPT_ARGS'):
        script = f'{script} {script_args}'
    path = f'{env["DIST_TRIAL_PATH"]}/output/out.{node}.txt'
    async for _ in (await asyncio.sleep(0.01 * i) 
        for i in range(50) if 'DIST_RESUME_PATH' not in env):
        if not os.path.exists(path): break
    logging.info(f'launch task {envs} in node {host}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f: 
        f.write(f'Launch with {env}\n {envs} {script}\n')
    popen = subprocess.Popen(f'{envs} {script}', env=env, 
        shell=True, start_new_session=True,
    stdout=open(path, 'a'), stderr=subprocess.STDOUT)
    t = asyncio.create_task(register_dist_task(env, popen))
    t.add_done_callback(lambda _: t.result())

async def register_dist_task_(env: dict, url: str) -> bool:
    try: return await request(cached_session().post, url, 
    ssl=False, timeout=60, json=env, allow_redirects=True)
    except asyncio.CancelledError: return False
    except: await asyncio.sleep(10); return True

async def register_dist_task(env, popen: subprocess.Popen):
    url = f'http://{CLUSTER}/dist/task/register'
    async def async_check_when():
        while popen.poll() is None: await asyncio.sleep(0.1)
    async_check = asyncio.create_task(async_check_when())
    while r := (await asyncio.wait([async_check,
    asyncio.create_task(register_dist_task_(env, url))], 
    return_when=asyncio.FIRST_COMPLETED
        ))[0].pop().result(): pass
    env = {**env, 'DIST_REGISTER_TIMEOUT': '0',
    'DIST_TASK_STATUS': 'FAILED' if popen.poll() else 'SUCCESS'}
    try: os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
    except ProcessLookupError: pass
    await asyncio.wait_for(async_check_when(), 10)
    if r is None: await register_dist_task_(env, url)

def wait_all_process(): os.wait(); wait_all_process()
SIGS = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
def handle_exit_signal(*_): 
    try: os.system(f'pkill -P {os.getpid()}'); wait_all_process()
    except OSError: os._exit(0)
import atexit; atexit.register(handle_exit_signal)
for sig in SIGS: signal.signal(sig, handle_exit_signal)

async def register_dist_node(env: dict, device_kind: str):
    for s in SIGS: asyncio.get_running_loop().add_signal_handler(
    s, lambda: asyncio.create_task(handle_async_exit()))
    try: gpu_kind, gpu = parse_gpu_kind_for_nvidia_device()
    except: gpu_kind, gpu = None, None
    device_kind = device_kind or gpu_kind or 'CPU'
    data = ['', env, [device_kind, gpu, os.cpu_count()], None, {}]
    url = f'http://{CLUSTER}/dist/node/register'
    while data[2]: data = await register_dist_node_(url, data)

async def handle_async_exit(timeout: float=1.0):
    cancel_coros = ['register_dist_node', 'register_dist_task_']
    for t in (tasks := [t for t in asyncio.all_tasks() if 
        t.get_coro().__name__ in cancel_coros]): t.cancel()
    res = await asyncio.wait_for(asyncio.gather(*tasks, 
        return_exceptions=True), timeout)
    logging.info(f'exit with {res}'); handle_exit_signal()

async def register_dist_node_(url: str, data: list) -> list:
    try: data = await request(cached_session().post, url, 
    ssl=False, timeout=60, json=data, allow_redirects=True)
    except asyncio.CancelledError: data[2][0] = None; return data
    except: await asyncio.sleep(10); return data
    for k, v in data[-2].items():
         ROUTERS.update({k:v}) if v else ROUTERS.pop(k, None)
    if not (task := data[-1].copy()): return data
    try: await launch_dist_task(data[0], task)
    except Exception as e: 
        logging.warning(f'launch task {task} err, {e}')
    logging.info(f'launch task with env {task}'); return data

def parse_gpu_kind_for_nvidia_device() -> tuple:
    dir = '/proc/driver/nvidia'; gpus = os.listdir(f'{dir}/gpus')
    info_path = f'{dir}/gpus/{gpus[0]}/information'
    parse = lambda f: f.readline().split(' ')[-1].strip()
    with open(info_path) as f: return parse(f), len(gpus)

if __name__ == '__main__': 
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')

logging.basicConfig(filename='./manager.log', level=logging.INFO)