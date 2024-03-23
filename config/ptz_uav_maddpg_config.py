from easydict import EasyDict
import sys

sys.path.append('D:/DI-engine/UAV')
n_agent = 2
n_landmark = 2
n_user = 20
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='3.23.11.39_TD3_cycle30_scale+0.01',
    env=dict(
        env_family='mpe',
        env_id='uav_env_v0',
        n_agent=n_agent,
        n_landmark=n_landmark,
        n_user=n_user,
        max_cycles=30,
        agent_obs_only=False,
        agent_specific_global_state=True,
        continuous_actions=True,  # ddpg only support continuous action space
        act_scale=True,  # necessary for continuous action space
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=0,
        rendermode='rgb_array',  # rgb_array or human
        flag_plot = False # 是否绘制图像
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        random_collect_size=5000,
        model=dict(
            agent_obs_shape=2 + 2 + n_user + n_landmark * 2 + (n_agent - 1) * 2 + 2 * n_user,
            global_obs_shape=2 + 2 + n_user + n_landmark * 2 + (
                    n_agent - 1) * 2 + 2 * n_user + 4 * n_agent + 2 * n_landmark + n_user + 2 * n_user,
            action_shape=5 + n_user * (n_landmark + 1),
            action_space='regression',
            twin_critic=True, # True for TD3
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            # learning_rates
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            target_theta=0.005,
            discount_factor=0.99,
        ),
        collect=dict(
            n_sample=1600,
            env_num=collector_env_num,
        ),
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=500, ),
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)

main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['envs.ptz_uav_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='td3'),
)
create_config = EasyDict(create_config)
ptz_simple_spread_maddpg_config = main_config
ptz_simple_spread_maddpg_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_entry -c ptz_simple_spread_maddpg_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0, max_env_step=int(4e6))
