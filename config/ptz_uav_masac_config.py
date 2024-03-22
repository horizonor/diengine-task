from easydict import EasyDict
import sys

sys.path.append('D:/DI-engine/UAV')

n_agent = 2
n_landmark = 2
n_user = 20
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='3.21_18.29_masac',
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
        flag_plot=False
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        multi_agent=True,
        random_collect_size=5000,
        model=dict(
            agent_obs_shape=2 + 2 + n_user + n_landmark * 2 + (n_agent - 1) * 2 + 2 * n_user,
            global_obs_shape=2 + 2 + n_user + n_landmark * 2 + (
                    n_agent - 1) * 2 + 2 * n_user + 4 * n_agent + 2 * n_landmark + n_user + 2 * n_user,
            action_shape=5 + n_user * (n_landmark + 1),
            action_space='reparameterization',
            twin_critic=True,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            # learning_rates
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            target_entropy=-2,
        ),
        collect=dict(
            n_sample=1600,
            env_num=collector_env_num,
        ),
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=50, ),
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
        import_names=['dizoo.petting_zoo.envs.petting_zoo_simple_spread_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac'),
)
create_config = EasyDict(create_config)
ptz_simple_spread_masac_config = main_config
ptz_simple_spread_masac_create_config = create_config
if __name__ == '__main__':
    # or you can enter `ding -m serial -c ptz_simple_spread_qmix_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
