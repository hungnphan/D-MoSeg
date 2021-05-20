#!/bin/bash

# Create template header for benchmark
echo -e "\nbatch_size,num_proc,threshold_value,num_messages,training_duration\n" >> benchmark_result.csv


threstype=0
# thresval=0.01
# batchsize=2
numepoch=10
logging=1
logfreq=50

batchsizes=( 4 8 12 )
numprocs=( 2 4 8 )
maxproc=( 20 12 8 )
thresvals=(
	0.0001 0.0002 0.0004 0.0006 0.0008 
	0.001 0.002 0.004 0.006 0.008 
	0.01 0.02 0.04 0.06 0.08 
	# 0.1 0.2 0.4 
	# 0.6 0.8
	# 1.0 2.0 10.0 20.0 50.0
)

# Iterate batchsizes
for i in "${!batchsizes[@]}"
do
	# Iterate numprocs
	for numproc in "${numprocs[@]}"
	do

		if [[ $numproc -gt ${maxproc[$i]} ]]
	   	then
	      	break
	   	fi

		for thresval in "${thresvals[@]}"
		do
			echo -e "\n---------------------------------------------------------------"
			echo "Running ${numproc} processes with ${numepoch} epoches: thresval=${thresval}, batch_sz=${batchsizes[$i]}"
			echo -e "---------------------------------------------------------------\n"

		upcxx-run -n ${numproc} example-app ${threstype} ${thresval} ${batchsizes[$i]} ${numepoch} ${logging} ${logfreq}
		done
		
	done
done
