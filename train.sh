#!/bin/bash
#SBATCH --job-name=test-slurm-gpu
#SBATCH --partition=ird_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=train.out
#SBATCH --error=train.err

# Charger les modules nécessaires
module () {
    eval `/usr/bin/modulecmd bash $*`
}

export PATH=$PATH:/usr/local/cuda/bin
export INCLUDE=$INCLUDE:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

module load bioinfo-cirad
module load pytorch/2.x

# Afficher quelques infos pour debug
nvcc --version
python3 --version

# Lancer le script Python
python3 main.py
