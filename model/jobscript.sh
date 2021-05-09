#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -t 0-24:00:00
#SBATCH -o job.out

# Load modules
module load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 TensorFlow/2.2.0-Python-3.7.4
module load OpenCV scikit-learn matplotlib

echo "> Modules loaded"

# Number of runs
RUNS=10

echo "> Running run_model.py ${RUNS} times"

# The program to run

for ((i=1; i<=$RUNS; i++))
do  
    echo "> Running ${i} program"
    python3 run_model.py
done

echo "> Counting accuracy for ${RUNS} runs:"

# Count accuracy
python3 average_accuracy.py
