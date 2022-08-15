#!/bin/bash

#SBATCH -n 16 # cores
#SBATCH -N 1  # nodes
#SBATCH -t 0-01:00:00 # max 7 days
#SBATCH -p cpu
#SBATCH -A venkvis
#SBATCH --mem-per-cpu=2280 # Memory pool for all cores in MB
#SBATCH -e log_%j.err
#SBATCH -o log_%j.out   # STDOUT; %j - job # 
#SBATCH --mail-type=ALL # email notif type- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=vedantpuri@cmu.edu

####SBATCH --exclude=f012,f020 # nodes to avoid

echo "Job started on `hostname` at `date`" 

module load miniconda3
source /home/vedantpu/.bash_profile
conda init bash
conda deactivate
conda activate FlowModels
python3 script.py

echo " "
echo "Job Ended at `date`"

### interactive
##srun -A venkvis -n 16 -N 1 --pty bash
