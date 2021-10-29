#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=4
#SBATCH --ntasks=81
#SBATCH --output=regression_out.%j
#SBATCH --error=regression_err.%j
#SBATCH --time=00:10:00
#SBATCH --mail-type=END
#SBATCH --partition=batch

module --force purge
module use $OTHERSTAGES

module load Stages/2020
module load GCC/9.3.0
module load OpenMPI/4.1.0rc1

module load Python/3.8.5
module load SciPy-Stack/2020-Python-3.8.5
module load scikit/2020-Python-3.8.5
module load TensorFlow/2.3.1-Python-3.8.5
module load mpi4py/3.0.3-Python-3.8.5

srun python regression.py