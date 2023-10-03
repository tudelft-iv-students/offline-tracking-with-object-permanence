#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

#SBATCH --output=slurm_test-%j.out​
#SBATCH --error=slurm_test-%j.err

# The default partition is the 'general' partition
#SBATCH --partition=cor,general 

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=medium

# The default run (wall-clock) time is 1 minute
#SBATCH --time=18:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1 (note: CPUs are always allocated to jobs per 2)
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
#SBATCH --cpus-per-task=4

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=30gb

# Request a GPU
#SBATCH --gres=gpu:v100:1

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

# Your job commands go below here

# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load miniconda/3.7
module load cuda/11.4
conda activate pgp

cd /tudelft.net/staff-umbrella/stanliu/code/pgp/PGP/

# Computations should be started with 'srun'. For example:

# srun python train.py -c configs/occlusion_train_v2.yml -r /tudelft.net/staff-umbrella/stanliu/data/nuScenes/ -d ./preprocess_graph_occ_v3 -o ./output/test_occ/ugru/v3_no_map_agg_b64 -n 50

srun python train.py -c configs/match_train_augment.yml -r /tudelft.net/staff-umbrella/stanliu/data/nuScenes/ -d ./preprocess_graph_match_v3 -o ./output/test_match/adjust_time/uniform_time -n 50
    
     #python train.py -c configs/raster_home_1.yml -r /tudelft.net/staff-umbrella/stanliu/data/nuScenes/ -d ./preprocess_home -o ./output/test_home_original_slow/ -n 16
#srun  --exclusive --ntasks=1 

# Your job commands go above here

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
conda deactivate
