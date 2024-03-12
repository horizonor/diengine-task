from easydict import EasyDict
import pytest
import numpy as np
import pettingzoo
from ding.utils import import_module
import sys
sys.path.append('D:/DI-engine/UAV')
from envs.ptz_uav_env import PettingZooEnv


@pytest.mark.envtest
class TestPettingZooEnv:

    def test_agent_obs_only(self):
        n_agent = 1
        n_landmark = 2
        n_user = 2
        env = PettingZooEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='uav_env_v0',
                    n_agent=n_agent,
                    n_landmark=n_landmark,
                    n_user=n_user,
                    max_step=100,
                    agent_obs_only=True,
                    continuous_actions=True,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        assert obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray), timestep.obs
            assert timestep.obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
            assert timestep.reward.dtype == np.float32
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_dict_obs(self):
        n_agent = 1
        n_landmark = 2
        n_user = 2
        env = PettingZooEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='uav_env_v0',
                    n_agent=n_agent,
                    n_landmark=n_landmark,
                    n_user=n_user,
                    max_step=100,
                    agent_obs_only=False,
                    continuous_actions=True,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for k, v in obs.items():
            print(k, v.shape)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, dict), timestep.obs
            assert isinstance(timestep.obs['agent_state'], np.ndarray), timestep.obs
            assert timestep.obs['agent_state'].shape == (
                n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2
            )
            assert timestep.obs['global_state'].shape == (
                n_agent * (2 + 2) + n_landmark * 2 + n_agent * (n_agent - 1) * 2,
            )
            assert timestep.obs['agent_alone_state'].shape == (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2)
            assert timestep.obs['agent_alone_padding_state'].shape == (
                n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2
            )
            assert timestep.obs['action_mask'].dtype == np.float32
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_agent_specific_global_state(self):
        n_agent = 1
        n_landmark = 2
        n_user = 2
        env = PettingZooEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='uav_env_v0',
                    n_agent=n_agent,
                    n_landmark=n_landmark,
                    n_user=n_user,
                    max_step=100,
                    agent_obs_only=False,
                    agent_specific_global_state=True,
                    continuous_actions=True,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for k, v in obs.items():
            print(k, v.shape)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, dict), timestep.obs
            assert isinstance(timestep.obs['agent_state'], np.ndarray), timestep.obs
            assert timestep.obs['agent_state'].shape == (
                n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2
            )
            assert timestep.obs['global_state'].shape == (
                n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) +
                n_landmark * 2 + n_agent * (n_agent - 1) * 2
            )
            assert timestep.obs['agent_alone_state'].shape == (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2)
            assert timestep.obs['agent_alone_padding_state'].shape == (
                n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2
            )
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
