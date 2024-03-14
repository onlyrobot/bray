import numpy as np

def gae(trajectory: list, bootstrap_value=0.0):
    """Generalized Advantage Estimation"""
    trajectory.append({"advantage": 0.0, "value": bootstrap_value})
    for i in reversed(range(len(trajectory) - 1)):
        t, next_t = trajectory[i], trajectory[i + 1]
        target_value = t["reward"] + 0.99 * next_t["value"]
        # 0.99: discount factor of the MDP
        delta = target_value - t["value"]
        # 0.95: discount factor of the gae
        advantage = delta + 0.99 * 0.95 * next_t["advantage"]
        t["advantage"] = np.array(advantage, dtype=np.float32)
    trajectory.pop(-1)  # drop the fake


def multi_head_gae(trajectory, predict_outputs, multi_head_value_config, done, done_reward):
    if done:
        bootstrap_value = np.zeros(multi_head_value_config['num_heads'])
        trajectory.append({"value": bootstrap_value, "reward":done_reward})
    else:
        bootstrap_value = predict_outputs['value']
        trajectory.append({"value": bootstrap_value, "reward":np.zeros(multi_head_value_config['num_heads'])})
    multi_head_advantage_array = np.zeros((multi_head_value_config['num_heads'], len(trajectory) - 1))
    for head_index, head_name in enumerate(multi_head_value_config['value_loss_heads']):
        gamma = multi_head_value_config['gamma'][head_name]
        lamb  = multi_head_value_config['lamb'][head_name]
        if gamma == 0 and lamb == 0:
            for i in range(len(trajectory) - 1):
                multi_head_advantage_array[head_index][i] = trajectory[i+1]['reward'][head_index] - trajectory[i]['value'][head_index]
        else:
            gae_advantage = 0

            for i in reversed(range(len(trajectory) - 1)):
                value = trajectory[i]['value'][head_index]
                next_value = trajectory[i+1]['value'][head_index]

                reward = trajectory[i+1]['reward'][head_index]

                delta = reward + gamma * next_value - value # r1 + gamma * v1 * (1 - d1) - v0
                gae_advantage = delta + gae_advantage * lamb
                multi_head_advantage_array[head_index][i] = gae_advantage
                # if abs(gae_advantage) > 1:
                    # print('gae_advantage', gae_advantage, 'reward', reward, 'next_value', next_value, 'value', value, 'count', count)

    for m in range(len(trajectory)-1):
        advantage = 0
        target_value = np.zeros(multi_head_value_config['num_heads'])
        for i, head_name in enumerate(multi_head_value_config['value_loss_heads']):
            target_value[i] = trajectory[m]['value'][i] + multi_head_advantage_array[i][m]
            if head_name in multi_head_value_config['policy_loss_heads']:
                advantage += multi_head_advantage_array[i][m]
            trajectory[m]['single_head_advantage_'+head_name] = multi_head_advantage_array[i][m]

        trajectory[m]['target_value'] = np.array(target_value, dtype=np.float32)
        trajectory[m]['advantage'] = np.array(advantage, dtype=np.float32)

    trajectory.pop(-1)  # drop the fake
