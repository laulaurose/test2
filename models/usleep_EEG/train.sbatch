#!/usr/bin/bash
#SBATCH --job-name=EEG
#SBATCH --output=EEG.%j.out
#SBATCH --error=EEG.%j.err
#SBATCH -p gpu,mignot
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -G 1
source /home/groups/mignot/laurose/miniconda3/etc/profile.d/conda.sh
conda activate u-sleep
ml cudnn/8.6.0.163
ut preprocess --out_path /scratch/users/laurose/models/usleep_EEG/processed_data.h5 --dataset_splits train_data val_data --overwrite
ut train --num_gpus=1 --overwrite --preprocess --initialize_from '/home/groups/mignot/laurose/miniconda3/envs/u-sleep/lib/python3.11/site-packages/usleep/models/u-sleep-1eeg-2.0/model/model/@epoch_8632_val_dice_0.78740.h5'

