#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts
OMP_NUM_THREADS=1 mpirun python run_experiment.py 2 4 0.001 0.001 5 0 --mpi
