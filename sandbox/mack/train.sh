#!/bin/bash -x
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --job-name="train"
#SBATCH --output=/home/jeonwons/scratch/slurm_output/slurm-%a.out
#SBATCH --open-mode="truncate"
#SBATCH --array=0-9

module load singularity
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# 0. Setting
echo $HOSTNAME
export CONTAINER_NAME=wsjeon-marl-dev-setting-master-zsh.simg
export PROJECT_DIR=PycharmProjects/multiagent-gail

# # 1. Copy your container on the compute node
# rsync -avz $SCRATCH/singularity-images/$CONTAINER_NAME $SLURM_TMPDIR

# 2. Copy your dataset on the compute node, e.g., expert trajectories

# 3. Executing your code with singularity

singularity exec --nv \
        -H $HOME:/home \
        -B $SLURM_TMPDIR:/dataset/ \
        -B $SCRATCH:/tmp_log/ \
        -B $SCRATCH:/final_log/ \
        $SCRATCH/singularity-images/$CONTAINER_NAME \
        python -u $PROJECT_DIR/sandbox/mack/train_with_taskid.py --slurm_task_id=$SLURM_ARRAY_TASK_ID

# singularity exec --nv \
#         -H $HOME:/home \
#         -B $SLURM_TMPDIR:/dataset/ \
#         -B $SLURM_TMPDIR:/tmp_log/ \
#         -B $SCRATCH:/final_log/ \
#         $SLURM_TMPDIR/$CONTAINER_NAME \
#         python $PROJECT_DIR/experiments/train_with_taskid.py --slurm-task-id=$SLURM_ARRAY_TASK_ID

## 4. Copy whatever you want to save on $SCRATCH
#mkdir -p $SCRATCH/ray_results
#rsync -avz $HOME/ray_results $SCRATCH/ray_results
