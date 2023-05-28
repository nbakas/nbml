#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --cpus-per-task=128                # number of cores per task
#SBATCH --time=00:30:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=${account_name}          # project account
#SBATCH --qos=dev


module load env/staging/2022.1
module load Python/3.10.4-GCCcore-11.3.0
ml SciPy-bundle/2022.05-foss-2022a 
ml PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
ml IPython/8.5.0-GCCcore-11.3.0

cd nbml
python __nbml__.py
