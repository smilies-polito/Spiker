#!/usr/bin/bash

VHD="./Hierarchical/Vhd/Vhd"
FLAT="./Flat"

# Complete accelerator
cp "$VHD/CompleteAccelerator/complete_accelerator.vhd" $FLAT

# Input interface
cp "$VHD/InputInterface/ff.vhd" $FLAT
cp "$VHD/InputInterface/input_interface.vhd" $FLAT
cp "$VHD/InputInterface/lfsr.vhd" $FLAT
cp "$VHD/InputInterface/shift_register_unsigned.vhd" $FLAT
cp "$VHD/InputInterface/unsigned_cmp_gt.vhd" $FLAT

# Input buffer
cp "$VHD/InputBuffer/input_buffer.vhd" $FLAT
cp "$VHD/InputBuffer/reg_unsigned.vhd" $FLAT

# Output interface
cp "$VHD/OutputInterface/out_interface.vhd" $FLAT
cp "$VHD/OutputInterface/out_interface_cu.vhd" $FLAT
cp "$VHD/OutputInterface/out_interface_datapath.vhd" $FLAT

# Output selector
cp "$VHD/OutputSelector/generic_mux.vhd" $FLAT

# Spiker
cp "$VHD/spiker.vhd" $FLAT

# Layer
cp "$VHD/Layer/anticipate_bits.vhd" $FLAT
cp "$VHD/Layer/bit_selection.vhd" $FLAT
cp "$VHD/Layer/cmp_eq.vhd" $FLAT
cp "$VHD/Layer/generic_mux_1bit.vhd" $FLAT
cp "$VHD/Layer/generic_or.vhd" $FLAT
cp "$VHD/Layer/layer.vhd" $FLAT
cp "$VHD/Layer/layer_cu.vhd" $FLAT
cp "$VHD/Layer/layer_datapath.vhd" $FLAT
cp "$VHD/Layer/mux2to1.vhd" $FLAT
cp "$VHD/Layer/reg.vhd" $FLAT

# Complete neurons
cp "$VHD/CompleteNeurons/complete_neurons.vhd" $FLAT
cp "$VHD/CompleteNeurons/double_and_mask_n.vhd" $FLAT
cp "$VHD/CompleteNeurons/simple_and_mask.vhd" $FLAT

# Bare neurons
cp "$VHD/BareNeurons/bare_neurons.vhd" $FLAT
cp "$VHD/BareNeurons/generic_and.vhd" $FLAT

# Neuron 
cp "$VHD/Neuron/add_sub.vhd" $FLAT
cp "$VHD/Neuron/cmp_gt.vhd" $FLAT
cp "$VHD/Neuron/mux2to1_signed.vhd" $FLAT
cp "$VHD/Neuron/mux4to1_signed.vhd" $FLAT
cp "$VHD/Neuron/neuron.vhd" $FLAT
cp "$VHD/Neuron/neuron_cu.vhd" $FLAT
cp "$VHD/Neuron/neuron_datapath.vhd" $FLAT
cp "$VHD/Neuron/reg_signed_sync_rst.vhd" $FLAT
cp "$VHD/Neuron/shifter.vhd" $FLAT

# Synapses
cp "$VHD/Synapse/weights_bram.vhd" $FLAT

# Shared components
cp "$VHD/Shared/cnt.vhd" $FLAT
cp "$VHD/Shared/decoder.vhd" $FLAT
cp "$VHD/Shared/mux2to1_std_logic.vhd" $FLAT
cp "$VHD/Shared/reg_signed.vhd" $FLAT
