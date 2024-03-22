from easydict import EasyDict
import sys
import os
from copy import deepcopy
from functools import partial

import torch
from easydict import EasyDict
from tensorboardX import SummaryWriter

from ding.config import read_config, compile_config
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, InteractionSerialEvaluator
sys.path.append('/UAV')

n_agent = 2
n_landmark = 2
n_user = 20
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='3.21_13.20_qmix',
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
    policy=dict(type='discrete_sac'),
)
create_config = EasyDict(create_config)
ptz_simple_spread_masac_config = main_config
ptz_simple_spread_masac_create_config = create_config
if __name__ == '__main__':
    seed = 4
    input_cfg = (main_config, create_config)
    env_setting = None
    # Please add your model path here.
    model_path = r'D:\DI-engine\UAV\config\3.18_15.31_TD3\ckpt\ckpt_best.pth.tar'

    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    # evaluator_env.enable_save_replay(replay_path='./simple_spread_mappo_eval/video')
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=None, enable_field=['learn', 'collect', 'eval', 'command'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    # eval the pretrained moddl
    stop, eval_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, 0)
    print(eval_info)
