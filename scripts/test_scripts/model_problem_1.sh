#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts
OMP_NUM_THREADS=1 mpirun python run_experiment.py 8 256 0.001 0.001 10 0 --mpi
OMP_NUM_THREADS=1 mpirun python run_experiment.py 16 256 0.001 0.001 10 0 --mpi
OMP_NUM_THREADS=1 mpirun python run_experiment.py 32 256 0.001 0.001 10 0 --mpi
