#!/bin/bash

bitWidthFile="bitWidths.py"
tmpFile="tmp.py"
mnistTest="mnistTest.py"

for i in {10..32}
do
	# Substitute the bitWidth value and store in temporary file
	sed s/"neuron_bitWidth.*"/"neuron_bitWidth	= $i"/\
	$bitWidthFile > $tmpFile

	# Remove previous file
	rm $bitWidthFile

	# Rename the temporary file
	mv $tmpFile $bitWidthFile

	echo -e "Bit-width: $i\n"
	python $mnistTest
	echo -e "\n"
done
