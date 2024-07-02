#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=32gb 
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160M \
    --tasks wmdp-translate \
    --device cuda:0 \
    --batch_size 8