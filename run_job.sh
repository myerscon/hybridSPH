#!/bin/bash
#SBATCH -J pysph_shocktube_001_gsph    # name of my job
#SBATCH -A nse           # name of slurm account to use
#SBATCH -p share         # name of partition/queue to use
#SBATCH -o saved_data/shocktube/pysph_shocktube_001_gsph/sim.out        # name of output file for batch script
#SBATCH -e saved_data/shocktube/pysph_shocktube_001_gsph/sim.err        # name of error file for batch script
#SBATCH -n 1             # number of tasks
#SBATCH --time=03:00:00  # time needed for job

# gather basic information, can be useful for troubleshooting
hostname
echo $SLURM_JOBID
showjob $SLURM_JOBID

# load modules needed for job


# run my job

python3 pysph_shocktube_batch02.py