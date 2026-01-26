from bray.template import verify_default_template

TEMPLATES: dict[str: tuple] = {
'default': (
    {'DIST_TASK_TYPE': 'RM'}, lambda **kwargs: {
    # 'DIST_CONDA_ENV': 'ms-swift-latest',
    # 'DIST_SCRIPT': 'swift/rm.sh', 
    'VERIFY': verify_default_template,}),
}