from easydict import EasyDict
import sys

sys.path.append('D:/DI-engine/UAV')
n_agent = 2
n_landmark = 2
n_user = 20
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='3.17_8.15_reward_scale_80000n2_20user',
    env=dict(
        env_family='mpe',
        env_id='uav_env_v0',
        n_agent=n_agent,
        n_landmark=n_landmark,
        n_user=n_user,
        max_cycles=25,
        agent_obs_only=False,
        agent_specific_global_state=True,
        continuous_actions=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=0,
        rendermode='rgb_array',  # rgb_array or human
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            agent_num=n_agent,
            agent_obs_shape=2 + 2 + n_user + n_landmark * 2 + (n_agent - 1) * 2 + 2 * n_user,
            global_obs_shape=2 + 2 + n_user + n_landmark * 2 + (
                    n_agent - 1) * 2 + 2 * n_user + 4 * n_agent + 2 * n_landmark + n_user + 2 * n_user,
            action_shape=5 + n_user * (n_landmark + 1),
        ),
        learn=dict(
            multi_gpu=False,
            epoch_per_collect=5,
            batch_size=3200,
            learning_rate=5e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            env_num=collector_env_num,
        ),
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=50, ),
        ),
        other=dict(),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['envs.ptz_uav_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)
ptz_simple_spread_mappo_config = main_config
ptz_simple_spread_mappo_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_onpolicy -c ptz_simple_spread_mappo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy

    serial_pipeline_onpolicy((main_config, create_config), seed=0)
