#!/bin/sh

#SBATCH -t 16:00:00
#SBATCH --output=err.log
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --mem=120000
#SBATCH -J V-PcOH8_dft
##SBATCH --qos=Long

export OMP_NUM_THREADS=$SLURM_NTASKS

hostname

module switch PrgEnv-cray PrgEnv-gnu

# path to the orca software
export PATH="/path/to/orca_5_0_3_linux_x86-64_shared_openmpi411/:$PATH"
export LD_LIBRARY_PATH="/path/to/orca_5_0_3_linux_x86-64_shared_openmpi411:$LD_LIBRARY_PATH"

#path to the openmpi library
export PATH="/path/to/openmpi/bin/:$PATH"
export LD_LIBRARY_PATH="/path/to/openmpi/lib/:$LD_LIBRARY_PATH"

export bindir="/path/to/orca_5_0_3_linux_x86-64_shared_openmpi411"

echo $bindir


$bindir/orca V-PcOH8_dft.inp > V-PcOH8_dft.log



