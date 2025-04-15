#!/bin/bash
#SBATCH --job-name="nc2zarr"
#SBATCH --time=04:00:00
#SBATCH --mem=64000
#SBATCH --account=hrcm
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --array=0-365

ARRAY_INDEX=${SLURM_ARRAY_TASK_ID}

python nc_to_zarr.py ${ARRAY_INDEX}

