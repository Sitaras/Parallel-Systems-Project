#!/bin/bash

# Which Queue to use, DO NOT CHANGE #
#PBS -q GPUq

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:05:00

# How many nodes and tasks per node, Example 2 nodes with 8 tasks each => 16 tasts #
#PBS -l select=1:ncpus=1:ngpus=1


# JobName #
#PBS -N myGPUJob

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
nvprof --unified-memory-profiling off ./cuda1gpu < input

# profile executable #
#nvprof ./jacobi_cuda < input
