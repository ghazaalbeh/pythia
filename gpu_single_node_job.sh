#!/bin/bash
#SBATCH --account=def-agullive
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=28
#SBATCH --mem=150G
#SBATCH --time=1-00:00
module load arch/avx512 StdEnv/2018.3
source /home/behboud/ENV/bin/activate
python tools/run.py --tasks vqa --datasets textvqa --model mymodel --config configs/vqa/textvqa/lorra.yml
