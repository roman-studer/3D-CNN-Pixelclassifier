#!/bin/bash
#SBATCH -t 1-5:00:00
#SBATCH -p performance
#SBATCH --gpus=1
#SBATCH --job-name=bio_hyperSN
#SBATCH --output=bio_hyperSN.out
#SBATCH --error=bio_hyperSN.err

echo "Pulling latest version of 3D-CNN repository repository"
cd /mnt/nas05/data01/biocycle/3D-CNN-Pixelclassifier/ || exit
git status
git fetch
git pull git@github.com:roman-studer/3D-CNN-Pixelclassifier.git
echo "Pull successful"

echo "Starting training of hyperSN model"
SINGULARITYENV_LC_ALL=C.UTF-8 \
SINGULARITYENV_LANG=C.UTF-8 \
SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY \
singularity exec -B /mnt/nas05/data01/biocycle/:/workspace \
--nv /mnt/nas05/data01/biocycle/containers/lightning \
bash -c "python3 /workspace/3D-CNN-Pixelclassifier/scripts/hyperSN_trainer.py"
echo "Training finished"
