# noqa: D212, D415
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v3` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle
from env.some.core import Agent, Landmark, World, User
from env.some.scenario import BaseScenario
from env.some.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from UAV import parameters as param
import math

CPU_cycle = 200
F_u = param.F_u
kappa = param.Kappa
B_k = param.B_k
P_nt = 1 #TODO ：发射功率也要添加进行动里，暂时为1
sigma_e_2 = param.Sigma_e_2
g_0 = param.g_0
F_e_k = param.F_e_k
w1, w2 = param.w1, param.w2
Phi_n = param.Phi_n # 无人机覆盖角
angle_radians = math.radians(Phi_n)

class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            N=1,
            local_ratio=1,# 已经删除相关代码
            max_cycles=25,
            continuous_actions=True,
            render_mode=None,
            num_user=1,
            num_station=1,
            flag_plot=False
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
                0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N = N, num_user = num_user, num_station = num_station,flag_plot=flag_plot)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.render_mode = 'rgb_array'
        self.metadata["name"] = "uav_env_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=1, num_user=1, num_station=1, flag_plot=False):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = N
        world.flag_plot = flag_plot
        num_landmarks = num_station
        global num_users
        num_users = num_user
        # 任务维度为地表个数+1，因为无人机需要的决策是：{自身执行.基站1执行.基站2执行....}
        world.dim_t = num_users * (num_landmarks + 1)
        world.collaborative = True
        world.max_distance = 0.35
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.coverage = 0.35
            # agent.coverage = agent.h * math.tan(angle_radians) #这里暂时简化为max_distance
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.is_station = True
            landmark.size = 0.01
            landmark.id = i
        # add users
        world.users = [User() for i in range(num_users)]
        for i, user in enumerate(world.users):
            user.name = "user %d" % i
            user.collide = False
            user.movable = False
            user.size = 0.02
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # random properties for users
        for i, user in enumerate(world.users):
            user.color = np.array([0.35, 0.25, 0.35])
        # set random initial states
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            else:
                agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.service_vector = np.zeros(len(world.users))


        for i, landmark in enumerate(world.landmarks):# 使得基站分布在两侧
            if i == 0:
                landmark.state.p_pos = np.array([-1, 0])
            if i == 1:
                landmark.state.p_pos = np.array([1, 0])
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, user in enumerate(world.users):
            if i < 10:
                user.state.p_pos = np_random.uniform(0.3, 0.7, world.dim_p)
            else:
                user.state.p_pos = np_random.uniform(-0.6, -0.2, world.dim_p) # 使得用户分布在两个基站的两侧

            user.state.p_vel = np.zeros(world.dim_p)

    # def benchmark_data(self, agent, world):  # 为经过环境培训的策略提供诊断数据（例如评估指标）
    #     rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     min_dists = 0
    #     for lm in world.landmarks:
    #         dists = [
    #             np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
    #             for a in world.agents
    #         ]
    #         min_dists += min(dists)
    #         rew -= min(dists)
    #         if min(dists) < 0.1:
    #             occupied_landmarks += 1
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #                 collisions += 1
    #     return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.coverage + agent2.coverage
        return True if dist < dist_min else False

    def is_crossborder(self, agent): # 判断是否越界
        return True if np.abs(agent.state.p_pos[0]) > 1 or np.abs(agent.state.p_pos[1]) > 1 else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        n1, n2, n3 = 0, 0, 0
        for a in world.agents:
            if self.is_collision(a, agent) and a != agent:
                n1 += 1
            if self.is_crossborder(a):
                n3 += 1
        for u in world.users:
            if u.served_by is None:
                n2 += 1 # 每次有未被服务的用户，n2+1
        if n1 != 0 or n2 != 0 or n3 != 0:
            # TODO
            rew -= 100000 * n1 + 80000 * n2 + 50000 * n3
        else:
            for agent in world.agents:
                E_G2A, T_G2A, E_UAV, T_UAV, E_A2G, T_A2G, T_EC, T_m, T_n, E_n,E_m = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                for user in agent.serving_users:
                    T_EC, T_A2G, E_A2G = 0, 0, 0
                    E_G2A, T_G2A = self.G2A_trans(agent, user)
                    E_UAV, T_UAV = self.UAV_computing(agent, user)
                    for landmark in world.landmarks:
                        E_A2Gi, T_A2Gi = self.A2G_trans(agent, user, landmark)
                        T_ECi = self.EC_computing(agent, user, landmark, num_users)
                        T_EC += T_ECi
                        T_A2G += T_A2Gi
                        E_A2G += E_A2Gi
                    T_m = (T_G2A + max(T_UAV,T_A2G+T_EC)) #单个用户的时间
                    E_m = E_G2A+ E_UAV + E_A2G
                    T_n += T_m
                    E_n += E_m
                rew -= w1 * E_n + w2 * T_n
        # add reward scaling
        reward_scale = 0.00001
        rew *= reward_scale
        return rew

    # def global_reward(self, world):
    #     rew = 0
    #     # for user in world.users:
    #     #     dists = [
    #     #         np.sqrt(np.sum(np.square(a.state.p_pos - user.state.p_pos)))
    #     #         for a in world.agents
    #     #     ]
    #     #     rew -= min(dists)
    #     # add new reward for G2A transmission xb
    #     for agent in world.agents:
    #         E_G2A, T_G2A, E_UAV, T_UAV, E_A2G, T_A2G, T_EC, T_m, T_n, E_n,E_m = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #         for user in agent.serving_users:
    #             T_EC, T_A2G, E_A2G = 0, 0, 0
    #             E_G2A, T_G2A = self.G2A_trans(agent, user)
    #             E_UAV, T_UAV = self.UAV_computing(agent, user)
    #             for landmark in world.landmarks:
    #                 E_A2Gi, T_A2Gi = self.A2G_trans(agent, user, landmark)
    #                 T_ECi = self.EC_computing(agent, user, landmark, num_users)
    #                 T_EC += T_ECi
    #                 T_A2G += T_A2Gi
    #                 E_A2G += E_A2Gi
    #             T_m = (T_G2A + max(T_UAV,T_A2G+T_EC)) #单个用户的时间
    #             E_m = E_G2A+ E_UAV + E_A2G
    #             T_n += T_m
    #             E_n += E_m
    #         rew += w1 * E_n + w2 * T_n
    #     return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        user_pos = []
        for user in world.users:
            user_pos.append(user.state.p_pos - agent.state.p_pos)
        landmark_pos = []
        for landmark in world.landmarks:
            landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)
        onehot_dim = np.array([1.0 if l is agent else 0.0 for l in world.agents])

        new_phy =  np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + [agent.service_vector] + landmark_pos + other_pos + user_pos
        )
        phy_feature = np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + other_pos + user_pos
        )
        return new_phy

    def G2A_trans(self, agent, user):
        # 从地面到天空的数据传输,返回计算能耗与时间延迟
        # 计算当前UAV与USER间的欧氏距离（无人机距离地面有self.h的高度，目前是固定的）：
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - user.state.p_pos))
                       + np.square(agent.h))
        # 计算当前UAV与USER间的传输能耗：UE_num代表当前UAV服务的UE数量
        UE_num = agent.num_serving_users
        if UE_num == 0:
            # 如果UE_num为0，那么设置能耗为一个默认值，或者跳过能耗的计算
            E_G2A = 0
            T_G2A = 0
        else:
            T_G2A = agent.state.t[user.name]['task_size'] / ((10 / UE_num) * np.log2(1 + (25 / (10 * dist ** 2))))
            E_G2A = (0.1 * agent.state.t[user.name]['task_size']) / (
                        (10 / UE_num) * np.log2(1 + (25 / (10 * dist ** 2))))
        return E_G2A, T_G2A

    def UAV_computing(self, agent, user):
        # 无人机的计算能耗
        UE_num = agent.num_serving_users
        if UE_num != 0:
            T_UAV = (agent.state.t[user.name]['task_distribution'][0] * agent.state.t[user.name][
                'task_size'] * CPU_cycle) / (F_u / UE_num)
            E_UAV = kappa * ((F_u / UE_num) ** 3) * T_UAV
        else:
            T_UAV, E_UAV = 0, 0
        return E_UAV, T_UAV

    def A2G_trans(self, agent, user, landmark):
        # 空对地传输能耗
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))
                       + np.square(agent.h))
        UE_num = agent.num_serving_users
        if UE_num == 0:
            # 如果UE_num为0，代表没有任务被发送
            E_A2G = 0
            T_A2G = 0
        else:
            T_A2G = agent.state.t[user.name]['task_distribution'][landmark.id + 1] * agent.state.t[user.name]['task_size'] / (B_k * np.log2(1 + ((g_0* P_nt) / (sigma_e_2 * (dist ** 2)))))
            E_A2G = P_nt * T_A2G
        return E_A2G, T_A2G

    def EC_computing(self,agent,user,landmark,num_users):
        # 基站计算能耗
        UE_num = agent.num_serving_users
        if UE_num != 0:
            T_EC = (agent.state.t[user.name]['task_distribution'][landmark.id + 1] * agent.state.t[user.name][
                'task_size'] * CPU_cycle) / (F_e_k / num_users)
        else:
            T_EC = 0
        return T_EC