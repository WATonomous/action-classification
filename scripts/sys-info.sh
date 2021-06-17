#!/bin/bash
#SBATCH --time=0-00:01:00 # time (DD-HH:MM:ss)
echo "------ Gathering System Info"
set -x
hostname
uname -a
whoami
id
pwd
ls -al
df -h
printenv
nvidia-smi
lscpu
lspci
lshw
set +x
echo "------ Done gathering System Info"

