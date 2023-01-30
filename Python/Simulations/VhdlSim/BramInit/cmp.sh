#!/bin/bash

VHDL_DIR="../../../../Vhdl"
PARAMETERS_DIR="$VHDL_DIR/Hierarchical/Sim/Parameters"

# Compare the single init file with all the single BRAM files.

# Loop over all the BRAM files
for i in {0..57}
do
	# Print the index of the current BRAM file
	echo "BRAM number $i"

	# Select the subportion of the single init file to compare
	sed -n $((i*784+1)),$(((i+1)*784))p "$PARAMETERS_DIR/weights.mem" > \
		tmp.mem

	# Compare the subportion with the single BRAM init file
	diff tmp.mem "$PARAMETERS_DIR/weights"$i".mem"
done

rm tmp.mem
