#!/bin/bash
 
#SBATCH --nodes=64                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=16
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --mem-per-cpu=6000
#SBATCH --time=70:00:00             # the max wallclock time (time limit your job will run)
 
#SBATCH --job-name=test_layer     # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_keil02@uni-muenster.de # your mail address


# set an output file
#SBATCH --output /scratch/tmp/USER/RBLOD/final/model_problem_2.dat

# run the application
module add GCC/8.2.0-2.31.1
module add OpenMPI/3.1.3
module add Python/3.7.2
module add SuiteSparse

sleep 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1

echo "Launching job:"
cd /home/USER/RBLOD/scripts/test_scripts
./model_problem_2.sh

if [ $? -eq 0 ]
then
    echo "Job ${SLURM_JOB_ID} completed successfully!"
else
    echo "FAILURE: Job ${SLURM_JOB_ID}"
fi


