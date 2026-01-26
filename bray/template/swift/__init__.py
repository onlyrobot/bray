from bray.template import verify_default_template

TEMPLATES: dict[str: tuple] = {
'default': ({}, lambda **kwargs: {
    'DIST_CONDA_ENV': kwargs['DIST_CONDA_ENV'] or 'ms-swift-latest',
    'DIST_RESUMABLE': 'value', 
    'DIST_SCRIPT': kwargs['DIST_SCRIPT'] or f'script/swift/{kwargs["DIST_TASK_TYPE"].lower()}.sh',
    'VERIFY': verify_default_template,}),
}