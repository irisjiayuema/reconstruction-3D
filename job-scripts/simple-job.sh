#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=15:0:0    
#SBATCH --mail-user=iris_ma999@hotmail.com
#SBATCH --mail-type=ALL

cd ~/$projects/reconstruction-3D
module purge
module load python/3.9 scipy-stack StdEnv/2020  gcc/9.3.0  cuda/11.4 opencv/4.6.0
source ~/py39/bin/activate

python papr_scade.py train --data_dir datasets/scannet/ --scene_id scene0758_00 --cimle_dir dump_1102_scene0758_sfmaligned_indv --ckpt_dir logs_scannet --expname scene758 --opt configs/nerfsyn/chair.yml