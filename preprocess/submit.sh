#!/bin/sh
### General options
###  specify queue --
#BSUB -q gpuv100
### Mulige gpu clusters gpuv100, gpuk80 og gpuk40
### -- Ask for number of cores
#BSUB -n 4
### -- Specify that the process should be run exclusively on a gpu
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name --
#BSUB -J Norm 
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### User email address
#BSUB -u s183921@student.dtu.dk
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo ./out/output_%J.out
#BSUB -eo ./out/error_%J.err
# -- end of LSF options --

#Load modules
module load python3

python3 ./preprocess_spraakbanken.py
