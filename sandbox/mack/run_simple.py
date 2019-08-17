import logging
import os
import gym
import make_env_v2 as make_env
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from sandbox.mack.acktr_multi_disc import learn
from sandbox.mack.policies import CategoricalPolicy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args(parser):
    # _____ Experiment Setting
    parser.add_argument("--exp-name", type=str, default="MACK_MPE_Release")
    parser.add_argument("--max-episode-len", type=int, default=50)
    parser.add_argument("--num-timesteps", type=int, default=1500000)

    # _____ Optimization
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sample-batch-size", type=int, default=250)
    parser.add_argument("--train-batch-size", type=int, default=1000)

    # _____ Checkpoint
    parser.add_argument("--local-dir", type=str, default="/final_log/ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--scenario", type=str, default="simple_spread",
                        help="MPE scenario",
                        choices=["simple", "simple_speaker_listener",
                                 "simple_crypto", "simple_push",
                                 "simple_tag", "simple_spread", "simple_adversary"])

    # _____ Parallelism
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)

    return parser.parse_args()


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

    logger.configure(logdir, format_strs=['json'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = CategoricalPolicy
    learn(policy_fn,
          env,
          seed,
          total_timesteps=int(num_timesteps * 1.1),
          nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu,
          lr=lr,
          ent_coef=0.00,
          identical=make_env.get_identical(env_id),
          log_interval=50,
          save_interval=int(num_timesteps/timesteps_per_batch),
          max_episode_len=max_episode_len)
    logger.Logger.CURRENT.close()
    env.close()


def main(args):
    args.exp_name += "-{}".format(args.exp_id)
    if args.slurm_task_id is not None:
        args.exp_name += "/{}".format(str(args.slurm_task_id))

    from datetime import datetime
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    logdir = os.path.join(args.local_dir, args.exp_name, str(args.slurm_task_id), "MACK_mpe_0" + timestr)

    train(logdir,
          args.scenario,
          args.num_timesteps,
          args.lr,
          args.train_batch_size,
          args.run_id,
          args.num_envs_per_worker,
          args.max_episode_len)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/final_log/PycharmProjects/marl-rllib")
    from train import get_parser
    parser = get_parser()
    args = parse_args(parser)
    main(args)
