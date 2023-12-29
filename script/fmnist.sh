#!/bin/bash

#SBATCH --job-name=lt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
MIXUP=$1
ALPHA=$2


# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/hy2611/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
--overlay /scratch/hy2611/dataset/tiny-imagenet-200.sqf:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main_wb.py --dataset fmnist -a resnet32 --imbanlance_rate 1 --beta 0.5 --lr 0.01 \
 --epochs 200 --loss ce --resample_weighting 0 --mixup ${MIXUP} --mixup_alpha ${ALPHA} --store_name ce_mx${MIXUP}a${ALPHA}
"
