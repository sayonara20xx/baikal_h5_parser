#!/bin/bash
for (( c=1; c<=${1}; c++ ))
do
	echo "${c}"
	./util/create_h5.sh ${c}
done
