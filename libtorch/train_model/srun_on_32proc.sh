#!/bin/bash

#SBATCH --job-name=test-sys             # create a short name for your job
#SBATCH --partition=dgx2q               # total number of tasks across all nodes
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=8                      # number of cores on the compute node, or total number of tasks across all nodes
#SBATCH --gres=gpu:8                    # request use of GPUs on compute nodes
#SBATCH --output=slurm.%x.%j.out        # Name of file for stdout, default is jobID

module restore cuda10.1-cudnn7.6-opencv3.4

nvidia-smi

threstype=0
numepoch=15

batchsizes=( 16 8 4 2 )
numproc=32
thresvals=(
	0.0
	0.00001 0.000025 0.00005 0.000075
	0.0001  0.00025  0.0005  0.00075
	0.001   0.0025   0.005   0.0075
	0.01    0.025    0.05    0.075
	0.1
)


# Iterate batchsizes
for i in "${!batchsizes[@]}"
do
	# if [[ $numproc -gt ${maxproc[$i]} ]]
   	# then
    #   	break
   	# fi

	for thresval in "${thresvals[@]}"
	do
		#echo -e "\n---------------------------------------------------------------"
		echo "Running ${numproc} processes with ${numepoch} epoches: thresval=${thresval}, batch_sz=${batchsizes[$i]}"
		#echo -e "---------------------------------------------------------------\n"

		upcxx-run -n ${numproc} example-app ${threstype} ${thresval} ${batchsizes[$i]} ${numepoch}
		#echo "upcxx-run -n ${numproc} example-app ${threstype} ${thresval} ${batchsizes[$i]} ${numepoch} ${logging} ${logfreq} 1"
	done
	
done

