#!/bin/bash

parallelismFile="parallelisms.py"
tmpFile="tmp.py"
mnistTest="mnistTest.py"

for i in {10..32}
do
	# Substitute the parallelism value and store in temporary file
	sed s/"neuron_parallelism.*"/"neuron_parallelism	= $i"/\
	$parallelismFile > $tmpFile

	# Remove previous file
	rm $parallelismFile

	# Rename the temporary file
	mv $tmpFile $parallelismFile

	echo -e "Parallelism: $i\n"
	python $mnistTest
	echo -e "\n"
done
