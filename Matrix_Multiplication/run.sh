#!/bin/bash
#$ -cwd
#$ -o "matrixmul.out"
#$ -e "matrixmulerr.out"
./build/mul 1024 128 -50 -30
