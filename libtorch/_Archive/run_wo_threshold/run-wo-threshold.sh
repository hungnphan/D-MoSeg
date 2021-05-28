#!/bin/bash

# Create template header for benchmark
echo -e "\nbatch_size,num_proc,threshold_value,num_messages,training_duration\n" >> benchmark_result.csv


threstype=0
# thresval=0.01
# batchsize=2
numepoch=5
logging=1
logfreq=20

batchsizes=( 4 8 12)
numprocs=( 2 4 8 12 16 20 24)
maxproc=( 20 12 8)
thresvals=(
	999999.0
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