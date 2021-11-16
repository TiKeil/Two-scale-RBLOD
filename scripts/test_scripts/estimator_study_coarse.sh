#!/bin/bash

cd ../..
source venv/bin/activate
cd scripts

# coarse estimator 

OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 1 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 2 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 3 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 4 --mpi --ces --uce 
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 5 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 6 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 7 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 8 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 9 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 10 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 11 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 12 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 13 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 14 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 15 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 16 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 17 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 18 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 19 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 20 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 21 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 22 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 23 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 24 --mpi --ces --uce
OMP_NUM_THREADS=1 mpirun python run_with_stored_stage_1.py 16 256 0.001 0.01 1 0 25 --mpi --ces --uce
