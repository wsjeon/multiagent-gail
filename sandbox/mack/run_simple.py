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
from sandbox.mack.acktr_multi_disc import learn
from sandbox.mack.policies import CategoricalPolicy

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, max_episode_len):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id, max_episode_len=max_episode_len)
            env.discrete_action_input = True
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    
    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = CategoricalPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=make_env.get_identical(env_id))
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='/tmp/atlas/u/lantaoyu')
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener',
                                          'simple_crypto', 'simple_push',
                                          'simple_tag', 'simple_spread', 'simple_adversary']), default='simple_push')
@click.option('--lr', type=click.FLOAT, default=0.1)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
@click.option('--atlas', is_flag=True, flag_value=True)
@click.option('--max_episode_len', type=click.INT, default=50)
def main(logdir, env, lr, seed, batch_size, atlas, max_episode_len):
    env_ids = [env]
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]
    max_episode_lens = [max_episode_len]

    print('logging to: ' + logdir)

    for env_id, seed, lr, batch_size, max_episode_len in itertools.product(env_ids, seeds, lrs, batch_sizes, max_episode_lens):
        train(logdir + '/exps/mack/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
              env_id, 5e7, lr, batch_size, seed, batch_size // 250, max_episode_len)


if __name__ == "__main__":
    main()
