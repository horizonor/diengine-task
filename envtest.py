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
sys.path.append('D:/DI-engine/UAV')

n_agent = 2
n_landmark = 2
n_user = 10
collector_env_num = 1
evaluator_env_num = 1
main_config = dict(
    exp_name='ptz_uav_mappo_eval_seed0',
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
        rendermode='human',  # rgb_array or human
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            agent_num=n_agent,
            agent_obs_shape=2 + 2 + n_user + n_landmark * 2 + (n_agent - 1) * 2 + 2 * n_user,
            global_obs_shape=2 + 2 + n_user + n_landmark * 2 + (n_agent - 1) * 2 + 2 * n_user + 4 * n_agent + 2 * n_landmark + n_user + 2 * n_user,
            action_shape=5 + n_user * (n_landmark + 1),
        ),
        learn=dict(
            multi_gpu=False,
            epoch_per_collect=5,
            batch_size=3200,
            learning_rate=8e-4,
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
    seed = 2
    input_cfg = (main_config, create_config)
    env_setting = None
    # Please add your model path here.
    model_path = r'D:\DI-engine\UAV\config\(best)no_one_hot_5000n1+8000n2\ckpt\ckpt_best.pth.tar'

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
