#!/usr/bin/bash
#SBATCH --job-name=pred_all
#SBATCH --output=pred_all.%j.out
#SBATCH --error=pred_all.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu,mignot
#SBATCH -c 4
#SBATCH -G 1
source /home/groups/mignot/laurose/miniconda3/etc/profile.d/conda.sh
conda activate u-sleep
ml cudnn/8.6.0.163
python main_pred_v2_aggregated.py 2 "/scratch/users/laurose/experiments/usleep-all/output/" 

