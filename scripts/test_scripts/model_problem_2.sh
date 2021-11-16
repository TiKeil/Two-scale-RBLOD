#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts
OMP_NUM_THREADS=1 mpirun python run_experiment.py 64 8192 0.01 0.0001 10 1 --mpi --oc --sc --sld
