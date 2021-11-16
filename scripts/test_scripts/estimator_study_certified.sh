#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts

# certified estimator

OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 1 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 2 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 3 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 4 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 5 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 6 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 7 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 8 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 9 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 10 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 11 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 12 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 13 --mpi --ces
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 14 --mpi --ces
