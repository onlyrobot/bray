from bray.template import verify_default_template

TEMPLATES: dict[str: tuple] = {
'default': (
    {'DIST_TASK_TYPE': 'SFT'}, lambda **kwargs: {
    # 'DIST_CONDA_ENV': 'ms-swift-latest',
    # 'DIST_SCRIPT': 'swift/sft.sh', 
    'VERIFY': verify_default_template,}),
}