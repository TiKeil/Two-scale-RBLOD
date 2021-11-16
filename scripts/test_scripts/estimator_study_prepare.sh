#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts
OMP_NUM_THREADS=1 mpirun -n 20 python run_experiment.py 16 256 0.001 0.000000001 1 0 --mpi --ces --p
