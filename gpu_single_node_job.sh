#!/bin/bash
#SBATCH --account=def-agullive
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=1-00:00
module load arch/avx512 StdEnv/2018.3
source /home/behboud/ENV/bin/activate
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yml
