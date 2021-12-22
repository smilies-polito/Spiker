#!/bin/bash

parallelismFile="parallelisms.py"
tmpFile="tmp.py"
mnistTest="mnistTest.py"
paramDir="./parameters"
accuracyFile=$paramDir"/testPerformance.txt"
extension=".txt"
errorFile=$paramDir"/errors.txt"


minDecimalBits=2
maxDecimalBits=10
fixedDecimalBits=2

minNeuronParallelism=5
maxNeuronParallelism=32
fixedIntegerParallelism=13

# Remove error file before starting: mnistTest.py works in append on it
if [[ -f $errorFile ]]
then
	rm $errorFile
fi

# Set the number of decimal bits to a fixed amount
sed s/"fixed_point_decimals.*"/"fixed_point_decimals	= $fixedDecimalBits"/ \
$parallelismFile > $tmpFile

# Remove previous file
rm $parallelismFile

# Rename the temporary file
mv $tmpFile $parallelismFile


for (( i = $minNeuronParallelism; i < $maxNeuronParallelism; i++ ))
do
	# Substitute the parallelism value and store in temporary file
	sed s/"neuron_parallelism.*"/"neuron_parallelism	= $i"/\
	$parallelismFile > $tmpFile

	# Remove previous file
	rm $parallelismFile

	# Rename the temporary file
	mv $tmpFile $parallelismFile

	echo -e "Parallelism: $i\n"
	echo -e "--------------------------------------------------\n"
	python $mnistTest
	echo -e "\n"
done




for (( i = $minDecimalBits; i < $maxDecimalBits; i++ ))
do
	# Substitute the decimal bits value and store in temporary file
	sed s/"fixed_point_decimals.*"/"fixed_point_decimals	= $i"/ \
	$parallelismFile > $tmpFile

	# Remove previous file
	rm $parallelismFile

	# Rename the temporary file
	mv $tmpFile $parallelismFile

	neuronParallelism=$(( $fixedIntegerParallelism + $i  ))

	# Set the parallelism to keep the integer part at a fixed length
	sed s/"neuron_parallelism.*"/"neuron_parallelism\
	= $neuronParallelism"/ $parallelismFile > $tmpFile

	# Remove previous file
	# rm $parallelismFile

	# # Rename the temporary file
	# mv $tmpFile $parallelismFile

	echo -e "Decimal bits: $i\n"
	echo -e "--------------------------------------------------\n"
	python $mnistTest
	mv $accuracyFile "${accuracyFile%%$extension}""_"$i"_bits""$extension"
	echo -e "\n"
done
