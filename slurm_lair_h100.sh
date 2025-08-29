#!/bin/bash

#SBATCH --account=cogneuroai
#SBATCH -J full-brrr
#SBATCH --gres=gpu:H100:1
#SBATCH --partition general
#SBATCH --output=./logs/full/%j.txt
#SBATCH --error=./logs/full/%j.err
#SBATCH --mail-type=None
#SBATCH --mail-user=dicksonb@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

echo "Python: $(which python)"

python -c "print('*'*50)"

#export OMP_NUM_THREADS=16

num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

# Calculate a unique port number based on the array task ID
# Start from base port 29500
port=$((27500 + SLURM_ARRAY_TASK_ID))

#torchrun --standalone --nproc_per_node=$num_gpus --master_port=$port train_batch.py $(sed -n "$SLURM_ARRAY_TASK_ID p" job_array_configs_lair_h100.txt)
bash run_ollama_models.sh

python -c "print('*'*50)"
