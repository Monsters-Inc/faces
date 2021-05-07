#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:2  # We're launching 2 nodes with 2 Nvidia T4 GPUs each
#SBATCH -t 0-24:00:00

python3 runmodel_onehot.py

#Here you should typically call your GPU-hungry application