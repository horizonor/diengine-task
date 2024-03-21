import numpy as np
import matplotlib.pyplot as plt

class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None
        # xb task info
        self.t = {}


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None
        # xb task action
        self.t = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 1.5
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()
        self.id = 0
        # task_info
        self.task_info = None
        self.task_size = None
        self.task_quantity = None
        # tasks
        self.tasks = []

class User(Entity):  # properties of user entities
    def __init__(self):
        super().__init__()
        # user is not served by any agent
        self.served_by = None
        # task_info
        self.task_size = None
        self.task = True
    def update_task(self):
        self.task = True #还原任务标志
        # Update the task size with a random value
        self.task_size = np.random.randint(1, 5)  # Mbits size_of_input_data
class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        # new add
        self.serving_users = []
        self.service_vector = None  # 二元服务向量
        self.h = 0.1  # meters altitude 无人机高度
        # task_parts
        self.task_parts = {}
        self.coverage = 0 # 无人机覆盖范围
    def assign_task(self, users):
        self.task_parts = {} # Reset the task parts
        for user_name in self.serving_users:
            user = next(user for user in users if user.name == user_name.name)
            if user.task:  # If the user has a task
                self.task_parts[user.name] = user.task_size  # 被服务的用户将当前的任务发送给无人机，以user_id:task_size的形式存储在task_parts中
                user.task = False  # Change the user's task flag


    def serve_user(self, users, max_distance):
        self.serving_users = [] # Reset the list of serving users
        # 初始化服务向量
        self.service_vector = np.zeros(len(users))
        for i, user in enumerate(users):
            if user.served_by is not None and user.served_by != self:
                continue  # Skip if the user is already served by another agent
            dist = np.sqrt(np.sum(np.square(self.state.p_pos - user.state.p_pos)))  # Calculate the distance
            if dist <= max_distance:
                self.serving_users.append(user)
                user.served_by = self
                self.service_vector[i] = 1

    @property
    def num_serving_users(self):
        return len(self.serving_users)  # Return the number of serving users



class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.users = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # xb task dimensionality
        self.dim_t = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.3
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        # 记录图片编号
        self.counter = 0
        self.flag_plot = False

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.users

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        t_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force, t_force= self.apply_action_force(p_force, t_force)
        # apply environment forces(暂无环境影响)
        # p_force = self.apply_environment_force(p_force)
        # integrate physical state and task state
        self.integrate_state(p_force, t_force)
        # update agent state

        if self.flag_plot:
            filename = f'task_distribution_{self.counter}.png'
            self.plot_task_distribution_together(self.agents, filename)
            self.counter += 1

    # gather agent action forces
    def apply_action_force(self, p_force, t_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
                t_force[i] = agent.action.t
                # arr = agent.action.t[:len(self.landmarks)+1]
                # t_force[i] = np.exp(arr) / np.sum(np.exp(arr))
                # agent.action.t = agent.action.t[len(self.landmarks)+1:]

        return p_force, t_force

    # gather physical forces acting on entities
    # def apply_environment_force(self, p_force):
    #     # simple (but inefficient) collision response
    #     for a, entity_a in enumerate(self.entities):
    #         for b, entity_b in enumerate(self.entities):
    #             if b <= a:
    #                 continue
    #             [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
    #             if f_a is not None:
    #                 if p_force[a] is None:
    #                     p_force[a] = 0.0
    #                 p_force[a] = f_a + p_force[a]
    #             if f_b is not None:
    #                 if p_force[b] is None:
    #                     p_force[b] = 0.0
    #                 p_force[b] = f_b + p_force[b]
    #     return p_force

    def integrate_state(self, p_force, t_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )
            # TODO 更新任务状态
            if entity.task_parts:  # If task_parts is not empty
                for user_name, task_size in entity.task_parts.items():
                    arr = t_force[i][:len(self.landmarks) + 1]
                    arr1 = np.exp(arr) / np.sum(np.exp(arr))
                    t_force[i]= t_force[i][len(self.landmarks)+1:]
                    entity.state.t[user_name] = {
                        'task_size': task_size,
                        'task_distribution': arr1
                    }
            else:  # If task_parts is empty
                entity.state.t = {}

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # 绘制任务分布图
    def plot_task_distribution_together(self, agents, filename):
        colors = ['r', 'g', 'b']
        bar_width = 0.35

        fig, axs = plt.subplots(1, len(agents))  # 创建一个1行，len(agents)列的子图

        for i, agent in enumerate(agents):
            # 获取用户和任务分布数据
            users = list(agent.state.t.keys())
            task_distributions = [agent.state.t[user]['task_distribution'] for user in users]

            for j, task_distribution in enumerate(task_distributions):
                axs[i].bar(j, task_distribution[0], color=colors[0], width=bar_width)
                axs[i].bar(j, task_distribution[1], bottom=task_distribution[0], color=colors[1], width=bar_width)
                axs[i].bar(j, task_distribution[2], bottom=task_distribution[0] + task_distribution[1], color=colors[2],
                           width=bar_width)

            # 只绘制用户名称后面的数字部分
            user_numbers = [user.split(' ')[1] for user in users]
            axs[i].set_xticks(np.arange(len(users)))
            axs[i].set_xticklabels(user_numbers)
            axs[i].set_title(agent.name)  # 添加标题

        plt.savefig(filename)