#!/bin/bash

# Which Queue to use,
#PBS -q GPUq

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:01:00

# How many nodes and tasks per node, Example 1 nodes with 1 GPU.
#PBS -l select=1:ncpus=1:ngpus=1 

# JobName #
#PBS -N myGPUJob

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
./dot-product

# profile executable #
#nvprof ./dot-product
