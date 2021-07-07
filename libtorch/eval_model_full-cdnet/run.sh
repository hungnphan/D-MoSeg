#!/bin/bash

# Create template header for benchmark
#echo -e "\nbatch_size,num_proc,threshold_value,num_messages,training_duration\n" >> benchmark_result.csv


threstype=0
numepoch=5
logging=1
logfreq=20

batchsizes=( 4 )
numprocs=( 2 4 8 16 )
thresvals=(
	0.0
	0.0001 0.00025 0.0005 0.00075
	0.001  0.0025  0.005  0.0075
	0.01   0.025   0.05   0.075
	0.1
)

# Iterate batchsizes
for i in "${!batchsizes[@]}"
do
	# Iterate numprocs
	for numproc in "${numprocs[@]}"
	do
		for thresval in "${thresvals[@]}"
		do
			#echo -e "\n---------------------------------------------------------------"
			#echo "Running ${numproc} processes with ${numepoch} epoches: thresval=${thresval}, batch_sz=${batchsizes[$i]}"
			#echo -e "---------------------------------------------------------------\n"

			./example-app ${numproc} ${thresval} ${batchsizes[$i]} 0 16
			
			#echo "./example-app ${numproc} ${thresval} ${batchsizes[$i]} 0 16"
			
			#upcxx-run -n ${numproc} example-app ${threstype} ${thresval} ${batchsizes[$i]} ${numepoch} ${logging} ${logfreq} 1
			#echo "upcxx-run -n ${numproc} example-app ${threstype} ${thresval} ${batchsizes[$i]} ${numepoch} ${logging} ${logfreq} 1"
		done
		
	done
done
