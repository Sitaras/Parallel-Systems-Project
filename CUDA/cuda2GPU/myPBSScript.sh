#!/bin/bash

# Which Queue to use, DO NOT CHANGE #
#PBS -q GPUq

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:05:00

# How many nodes and tasks per node, 1 node with 2 tasks/threads 2 GPU 
#PBS -lselect=1:ncpus=10:ompthreads=10:ngpus=2 -lplace=excl

# JobName #
#PBS -N myGPUJob

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# profile executable #
nvprof --print-summary-per-gpu ./cuda2gpu < input

