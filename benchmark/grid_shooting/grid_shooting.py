import random
from threading import Lock


class GridShooting:
    SIZE = 9
    MAX_CD = 5

    SHOOTING_REWARD = 10
    MOVE_REWARD = -1
    STAR_REWARD = 1

    def __init__(self, **kwargs):
        super(GridShooting, self).__init__(**kwargs)
        self.agents = {}
        self.reset()
        # step actions
        self.actions = {}
        self.mutex = None

    def reset(self):
        # agent = [x_pos, y_pos, shooting_cd]
        self.agents = {0: [0, 0, 0], 1: [self.SIZE - 1, self.SIZE - 1, 0]}

        self.star_idx = random.randint(0, 2)
        self.frame_no = 0

    def init(self):
        self.mutex = Lock()

    def star(self):
        # star position 
        if self.star_idx == 0:
            star_pos = [int(self.SIZE / 2), int(self.SIZE / 2)]
        elif self.star_idx == 1:
            star_pos = [0, self.SIZE - 1]
        else:
            star_pos = [self.SIZE - 1, 0]
        return star_pos

    # def state_space_info(self, model_id: int) -> StateSpaceInfo:
    #     return StateSpaceInfo(size=6 * self.SIZE + 2, range=None)

    # def action_space_info(self, model_id: int) -> ActionSpaceInfo:
    #     return ActionSpaceInfo(size=9, type=ActionType.DISCRETE, range=None)

    def get_state(self, team_id, member_id):
        state = []
        # agent position
        enemy_team_id = team_id ^ 1
        agents_pos = {}
        for id in range(2):
            agent_pos = [0. for _ in range(self.SIZE * 2)]
            agent_pos[self.agents[id][0]] = 1
            agent_pos[self.agents[id][1] + self.SIZE] = 1
            agents_pos[id] = agent_pos
        state.extend(agents_pos[team_id])
        state.extend(agents_pos[enemy_team_id])
        # star position
        star = self.star()
        star_pos = [0. for _ in range(self.SIZE * 2)]
        star_pos[star[0]] = 1
        star_pos[star[1] + self.SIZE] = 1
        state.extend(star_pos)
        # cd 
        state.append(self.agents[team_id][2] / float(self.MAX_CD))
        state.append(self.agents[enemy_team_id][2] / float(self.MAX_CD))
        if self.agents[team_id][2] > 0:
            legal_actions = {}
            for i in range(5):
                legal_actions[i] = 1
        else:
            legal_actions = None
        extra_info = {'legal_actions': legal_actions}
        return state, extra_info

    def check_shooting(self, team_id, shooting_action):
        enemy_team_id = 1 if team_id == 0 else 0
        shooting_action -= 5

        def check_pos(equal_idx, is_less_equal):
            compare_idx = 1 if equal_idx == 0 else 0
            if self.agents[team_id][equal_idx] == self.agents[enemy_team_id][equal_idx]:
                if is_less_equal and self.agents[team_id][compare_idx] <= self.agents[enemy_team_id][compare_idx]:
                    return True
                if not is_less_equal and self.agents[team_id][compare_idx] >= self.agents[enemy_team_id][compare_idx]:
                    return True
            return False

        if shooting_action == 0:
            return check_pos(1, True)
        elif shooting_action == 1:
            return check_pos(0, True)
        elif shooting_action == 2:
            return check_pos(1, False)
        elif shooting_action == 3:
            return check_pos(0, False)

        return False

    def move(self, team_id, action):
        if action == 0:
            return
        if action == 1:
            self.agents[team_id][0] = min(self.agents[team_id][0] + 1, self.SIZE - 1)
        elif action == 2:
            self.agents[team_id][1] = min(self.agents[team_id][1] + 1, self.SIZE - 1)
        elif action == 3:
            self.agents[team_id][0] = max(self.agents[team_id][0] - 1, 0)
        elif action == 4:
            self.agents[team_id][1] = max(self.agents[team_id][1] - 1, 0)
        # when use move, decrease cd
        self.agents[team_id][2] = max(self.agents[team_id][2] - 1, 0)

    def step(self, team_0_action, team_1_action):
        self.frame_no += 1
        epi_done = False
        rewards = [self.MOVE_REWARD, self.MOVE_REWARD]
        # first check shooting action
        if team_0_action >= 5:
            if self.agents[0][2] > 0:
                # team 0 can not use shooting
                team_0_action = 0
        if team_1_action >= 5:
            if self.agents[1][2] > 0:
                team_1_action = 0
        # team 0 use shooting
        if team_0_action >= 5:
            # hit the target
            if self.check_shooting(0, team_0_action):
                epi_done = True
                rewards[0] += self.SHOOTING_REWARD
                rewards[1] -= self.SHOOTING_REWARD
            # shooting cd
            self.agents[0][2] = self.MAX_CD
        # team 1 use shooting
        if team_1_action >= 5:
            # hit the target
            if self.check_shooting(1, team_1_action):
                epi_done = True
                rewards[0] -= self.SHOOTING_REWARD
                rewards[1] += self.SHOOTING_REWARD
            self.agents[1][2] = self.MAX_CD

        def calc_win_team(final_rewards):
            if final_rewards[0] == final_rewards[1]:
                return -1  # draw
            elif final_rewards[0] > final_rewards[1]:
                return 0  # team 0 win
            else:
                return 1

        # if done, return
        if epi_done:
            logs = []
            # for team_id in range(2):
            #     log = self._logpack()
            #     log.add_scalar("win", int(team_id == calc_win_team(rewards)))
            #     logs.append(log)
            return rewards, epi_done, logs

        # move action
        if team_0_action < 5:
            self.move(0, team_0_action)
        if team_1_action < 5:
            self.move(1, team_1_action)

        # check star reward
        recreate_star = False
        star = self.star()
        if star[0] == self.agents[0][0] and star[1] == self.agents[0][1]:
            recreate_star = True
            rewards[0] += self.STAR_REWARD
        if star[0] == self.agents[1][0] and star[1] == self.agents[1][1]:
            recreate_star = True
            rewards[1] += self.STAR_REWARD

        if recreate_star:
            candidates = [0, 1, 2]
            candidates.remove(self.star_idx)
            self.star_idx = random.choice(candidates)

        if self.frame_no >= 100:
            epi_done = True
        # return
        logs = []
        # for team_id in range(2):
        #     log = self._logpack()
        #     if epi_done:
        #         log.add_scalar("win", int(team_id == calc_win_team(rewards)))
        #     logs.append(log)
        return rewards, epi_done, logs
