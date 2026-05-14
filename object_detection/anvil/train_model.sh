#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=yolo_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chelleocampo@cpp.edu

module purge
module load anaconda
conda activate myenv

python train_model.py