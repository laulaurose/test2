#!/usr/bin/bash
#SBATCH --job-name=main
#SBATCH --output=main.%j.out
#SBATCH --error=main.%j.err
#SBATCH -p gpu,mignot
#SBATCH -G 1
cd /scratch/users/laurose/models/usleep_all/
sbatch /scratch/users/laurose/models/usleep_all/finetune.sbatch
cd /scratch/users/laurose/models/usleep_EEG/
sbatch /scratch/users/laurose/models/usleep_EEG/finetune.sbatch
cd /scratch/users/laurose/models/usleep_EMG/
sbatch /scratch/users/laurose/models/usleep_EMG/finetune.sbatch

