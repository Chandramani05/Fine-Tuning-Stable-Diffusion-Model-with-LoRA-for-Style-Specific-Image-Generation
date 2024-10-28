#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=train_lora

module purge
module load python/intel/3.8.6

singularity exec --nv --bind /scratch/chandramaniyadav --overlay /scratch/chandramaniyadav/ml_project_env/overlay-25GB-500K.ext3:ro /scratch/chandramaniyadav/ml_project_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif/ /bin/bash -c "

source /ext3/env.sh

pwd

cd /chandramaniyadav/Downloads/Stable Diffusion with LORA

accelerate launch train_lora_scribble.py

exit
"
