#!/bin/bash
for (( c=1; c<=${1}; c++ ))
do
	echo "${c}"
	./util/create_h5_dataset_ext.sh ${c}
done
