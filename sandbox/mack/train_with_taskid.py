import itertools
import os
from subprocess import call
import click


@click.command()
@click.option('--slurm_task_id', type=click.INT, default=0)
def main(slurm_task_id):
    env = [  # Difficult task first.
        "simple_tag",
        "simple_spread",
        "simple_speaker_listener",
        "simple_push",
        "simple_crypto",
        "simple_adversary",
        "simple",
    ]  # 7
    max_episode_len = [25, 50]  # 2
    # seed = [slurm_task_id]  # 10
    seed = list(range(10))

    # NOTE: Totally, 7 x 2 x 10 = 140 configurations.
    hyperparameters_list = list(itertools.product(max_episode_len,
                                                  env,
                                                  seed))
    hyperparameters = hyperparameters_list[slurm_task_id]
    max_episode_len, env, seed = hyperparameters
    logdir = os.path.join(os.environ['HOME'], "ray_results/MACK_MPE_v2")

    interpreter = "python -u "
    command = interpreter + os.path.join(os.environ['HOME'],
                                         "PycharmProjects/multiagent-gail/sandbox/mack/run_simple.py")

    strings = []
    for key in ["max_episode_len", "env", "seed", "logdir"]:
        command += ' --' + key + '={}'.format(str(eval(key)))

    print("+ {}".format(command))
    print("[In train_with_taskid.py] Learning starts.")
    call(command, shell=True, executable='/bin/bash')
    print("[In train_with_taskid.py] Learning ends.")

    # for hyperparameters in list(hyperparameters_list):
    #     max_episode_len, env, seed = hyperparameters
    #     logdir = os.path.join(os.environ['HOME'], "ray_results/MACK_MPE_v2")

    #     interpreter = "python -u "
    #     command = interpreter + os.path.join(os.environ['HOME'],
    #                                          "PycharmProjects/multiagent-gail/sandbox/mack/run_simple.py")

    #     strings = []
    #     for key in ["max_episode_len", "env", "seed", "logdir"]:
    #         command += ' --' + key + '={}'.format(str(eval(key)))

    #     print("+ {}".format(command))
    #     print("[In train_with_taskid.py] Learning starts.")
    #     call(command, shell=True, executable='/bin/bash')
    #     print("[In train_with_taskid.py] Learning ends.")


if __name__ == '__main__':
    main()
