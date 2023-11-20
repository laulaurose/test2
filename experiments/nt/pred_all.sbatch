#!/usr/bin/bash
#SBATCH --job-name=pred_ALL
#SBATCH --output=pred_ALL.%j.out
#SBATCH --error=pred_ALL.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu,mignot
#SBATCH -c 4
#SBATCH -G 1
source /home/groups/mignot/laurose/miniconda3/etc/profile.d/conda.sh
conda activate u-sleep
ml cudnn/8.6.0.163
python pred.py "/scratch/users/laurose/models/usleep_all/" "/scratch/users/laurose/usleep_all/nt_data/" "all" 2 "/scratch/users/laurose/experiments/nt/predictions/all/"
