#!/usr/bin/bash
#SBATCH --job-name=ft_all
#SBATCH --output=ft_all.%j.out
#SBATCH --error=ft_all.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu,mignot
#SBATCH -c 8
#SBATCH -G 1
source /home/groups/mignot/laurose/miniconda3/etc/profile.d/conda.sh
conda activate u-sleep
ml cudnn/8.6.0.163
ut train --num_gpus=1 --overwrite --preprocess --continue_training
