#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym

import make_env_v2 as make_env
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet
from irl.mack.gail import learn
from sandbox.mack.policies import CategoricalPolicy
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, max_episode_len, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id, max_episode_len=max_episode_len)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    print(num_cpu)
    policy_fn = CategoricalPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    learn(policy_fn, expert, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=make_env.get_identical(env_id))
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='/atlas/u/lantaoyu/exps')
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener', 'simple_crypto',
                                          'simple_push', 'simple_tag', 'simple_spread']))
@click.option('--expert_path', type=click.STRING,
              default='/atlas/u/lantaoyu/exps/mack/simple_tag/l-0.1-b-1000/seed-1/checkpoint20000-1000tra.pkl')
@click.option('--atlas', is_flag=True, flag_value=True)
@click.option('--seed', type=click.INT, default=1)
@click.option('--traj_limitation', type=click.INT, default=200)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
@click.option('--disc_type', type=click.Choice(['decentralized', 'centralized', 'single']), default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
@click.option('--max_episode_len', type=click.INT, default=50)
def main(logdir, env, expert_path, atlas, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters, max_episode_len):
    env_ids = [env]
    lrs = [0.1]
    seeds = [seed]
    batch_sizes = [1000]
    max_episode_lens = [max_episode_len]

    logdir = '/atlas/u/lantaoyu/exps'

    # if env == 'simple_spread':
    #     expert_path = '/atlas/u/lantaoyu/exps/mack/simple_spread/l-0.1-b-1000/seed-4/checkpoint11000.pkl'
    # elif env == 'simple_speaker_listener':
    #     expert_path = '/atlas/u/lantaoyu/exps/mack/simple_speaker_listener/l-0.1-b-500/seed-2/checkpoint22000.pkl'
    # elif env == 'simple_tag':
    #     expert_path = '/atlas/u/lantaoyu/exps/mack/simple_tag/l-0.1-b-1000/seed-1/checkpoint11000.pkl'

    for env_id, seed, lr, batch_size, max_episode_len in itertools.product(env_ids, seeds, lrs, batch_sizes, max_episode_lens):
        train(logdir + '/gail/' + env_id + '/' + disc_type + '/s-{}/l-{}-b-{}-d-{}-c-{}/seed-{}'.format(
              traj_limitation, lr, batch_size, dis_lr, bc_iters, seed),
              env_id, 5e7, lr, batch_size, seed, batch_size // 250, max_episode_len, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters)


if __name__ == "__main__":
    main()
