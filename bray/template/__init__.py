from importlib import import_module, reload

modules = ['swift.grpo', 'swift.rm', 'swift.sft', 'swift']

def build_templates(m: str) -> dict[str: tuple]: 
    module = reload(import_module(f'mllm.template.{m}'))
    ts = getattr(module, 'TEMPLATES')
    return {f'{m}.{k}': v for k, v in ts.items()}

def verify_default_template(**kwargs) -> str: pass
TEMPLATES: dict[str: tuple] = {'default': (
    {}, lambda **_: {'VERIFY': verify_default_template})}

def match_template(**kwargs) -> tuple[str, dict[str]]:
    match = lambda x: match_count(kwargs, x[1][0])
    matched = sorted(filter(lambda x: match(x) >= 0, 
        TEMPLATES.items()), key=match)
    ts = [t[1][1](**kwargs) for t in matched]
    template = {k: v for t in ts for k, v in t.items()}
    if kwargs == (new_kwargs := kwargs | template): 
        return matched[-1][0], template
    else: return match_template(**new_kwargs)

def match_count(origin: dict, target: dict) -> int:
    count = len({k: v for k, v in target.items() 
        if origin.get(k) == v})
    return -1 if count != len(target) else count

for m in modules: TEMPLATES.update(build_templates(m))