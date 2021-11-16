#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts
OMP_NUM_THREADS=1 mpirun python run_experiment.py 16 256 0.1 0.000000001 1 0 --mpi
OMP_NUM_THREADS=1 mpirun python run_experiment.py 16 256 0.01 0.000000001 1 0 --mpi
OMP_NUM_THREADS=1 mpirun python run_experiment.py 16 256 0.001 0.000000001 1 0 --mpi
OMP_NUM_THREADS=1 mpirun python run_experiment.py 16 256 0.0001 0.000000001 1 0 --mpi
