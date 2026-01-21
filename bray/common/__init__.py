import os, httpx, json, socket

def cached_session(flush=False) -> httpx.AsyncClient:
    if not flush and CACHED_SESSION: return CACHED_SESSION
    session = httpx.AsyncClient()
    set_session(session); return CACHED_SESSION

def set_session(s): global CACHED_SESSION; CACHED_SESSION = s
CACHED_SESSION: httpx.AsyncClient = None

async def request(method: str, url: str, **kwargs) -> object:
    r = await cached_session().request(method, url, **kwargs)
    if r.status_code == 200: return r.json()
    raise Exception(f'request {url} err: {r.reason_phrase}')

try: HOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror: HOST = 'localhost'

def get_trial_path(project: str, trial: str) -> str:
    return os.path.realpath(f'{get_project_path(project)}/{trial}')

def get_project_path(project: str='') -> str: 
    return os.path.join(os.getcwd(), f'trial/{project}')

def load_trial_config(project: str, trial: str) -> dict:
    trial_path = f'{get_trial_path(project, trial)}/config.json'
    with open(trial_path, 'r') as f: return json.load(f)

def save_trial_config(project: str, trial: str, env: dict):
    trial_path = f'{get_trial_path(project, trial)}/config.json'
    os.makedirs(os.path.dirname(trial_path), exist_ok=True)
    with open(trial_path, 'w') as f: json.dump(env, f, indent=4)

def load_task_config(project: str, trial: str) -> dict:
    task_path = get_task_path(project, trial)
    with open(task_path, 'r') as f: return json.load(f)

def save_task_config(project: str, trial: str, env: dict):
    task_path = get_task_path(project, trial)
    if not env and not os.path.exists(task_path): return
    elif not env: return os.remove(task_path)
    os.makedirs(os.path.dirname(task_path), exist_ok=True)
    with open(task_path, 'w') as f: json.dump(env, f, indent=4)

def get_output_path(project, trial, node: int) -> str:
    trial_path = get_trial_path(project, trial)
    return f'{trial_path}/output/out.{node}.txt'

def get_task_path(project: str, trial: str) -> str:
    return f'{get_trial_path(project, trial)}/output/task.json'