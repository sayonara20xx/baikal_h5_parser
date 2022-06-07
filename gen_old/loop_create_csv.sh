#!/bin/bash
for (( c=1; c<=${1}; c++ ))
do
	echo "${c}"
	./util/create_csv.sh ${c}
done
