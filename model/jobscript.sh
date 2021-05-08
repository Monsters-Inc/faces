#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -t 0-24:00:00
#SBATCH -o job.out

python3 run_model.py

