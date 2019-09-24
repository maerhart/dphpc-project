#!/bin/bash

# nodes = number of mpi processes (physical hosts): --nodes=1
# each node has a number of tasks = number of requiested cpu cores per process (e.g. for openmp or threading): --ntasks 1


# each node has a number gpu's: --gres=gpu:2 (min 1, max 4 (default if specify gpu:0))
srun --time=4:00:00 --partition=intelv100 --nodes=1 --ntasks 1 --gres=gpu:2 --hint=nomultithread --pty bash

# total number of gpu's: number of nodes * number of gpu per node

module load cuda openmpi
