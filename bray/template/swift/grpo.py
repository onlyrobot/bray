from bray.template import verify_default_template

TEMPLATES: dict[str: tuple] = {
'default': ({'DIST_TASK_TYPE': 'GRPO'}, lambda **kwargs: {
    'DIST_RESUMABLE': 'value', 
    # 'DIST_CONDA_ENV': 'ms-swift-latest',
    # 'DIST_SCRIPT': 'swift/grpo.sh', 
    'VERIFY': verify_default_template,}),
'DAPO': ({'DIST_TASK_TYPE': 'DAPO'}, lambda **kwargs: {
    'DIST_RESUMABLE': 'value', 
    'DIST_SCRIPT_ARGS': '--loss_type dapo --epsilon_high 0.28 '
    '--dynamic_sample true --max_resample_times 3 --overlong_filter true '
    '--soft_cache_length 4096 --soft_max_length 5000',
    'REWARDS': [['soft_overlong', '1', '', 'DAPO algo reward']],
    'VERIFY': verify_default_template,}),
'RLOO': ({'DIST_TASK_TYPE': 'RLOO'}, lambda **kwargs: {
    'DIST_RESUMABLE': 'value', 
    'DIST_SCRIPT_ARGS': '--advantage_estimator rloo --kl_in_reward true',
    'VERIFY': verify_default_template,}),
}