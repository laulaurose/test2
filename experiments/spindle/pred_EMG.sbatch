#!/usr/bin/bash
#SBATCH --job-name=pred_EMG
#SBATCH --output=pred_EMG.%j.out
#SBATCH --error=pred_EMG.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu,mignot
#SBATCH -c 4
#SBATCH -G 1
source /home/groups/mignot/laurose/miniconda3/etc/profile.d/conda.sh
conda activate u-sleep
ml cudnn/8.6.0.163
python pred.py "/scratch/users/laurose/models/usleep_EMG/" "/scratch/users/laurose/usleep_all/spindle_data/" "EMG" 1 "/scratch/users/laurose/experiments/spindle/predictions/EMG/"

