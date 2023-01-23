#!/usr/bin/bash

VHD_DIR="../../Vhd/Vhd"
DEBUG_DIR="../../Vhd/Debug"
TB_DIR="../../Tb/Debug/Layer"

TOP_ENTITY="debug_layer_tb"

# Neuron
xvhdl $VHD_DIR"/Neuron/add_sub.vhd"
xvhdl $VHD_DIR"/Neuron/cmp_gt.vhd"
xvhdl $VHD_DIR"/Neuron/mux2to1_signed.vhd"
xvhdl $VHD_DIR"/Neuron/mux4to1_signed.vhd"
xvhdl $VHD_DIR"/Neuron/neuron_cu.vhd"
xvhdl $VHD_DIR"/Neuron/neuron_datapath.vhd"
xvhdl $VHD_DIR"/Neuron/neuron.vhd"
xvhdl $DEBUG_DIR"/Neuron/debug_neuron_datapath.vhd"
xvhdl $DEBUG_DIR"/Neuron/debug_neuron.vhd"
xvhdl $VHD_DIR"/Neuron/reg_signed_sync_rst.vhd"
xvhdl $VHD_DIR"/Neuron/shifter.vhd"

# Bare neurons
xvhdl $DEBUG_DIR"/BareNeurons/debug_bare_neurons.vhd"
xvhdl $VHD_DIR"/BareNeurons/generic_and.vhd"

# Complete neurons
xvhdl $DEBUG_DIR"/CompleteNeurons/debug_complete_neurons.vhd"
xvhdl $VHD_DIR"/CompleteNeurons/double_and_mask_n.vhd"
xvhdl $VHD_DIR"/CompleteNeurons/simple_and_mask.vhd"

# Layer
xvhdl $VHD_DIR"/Layer/anticipate_bits.vhd"
xvhdl $VHD_DIR"/Layer/bit_selection.vhd"
xvhdl $VHD_DIR"/Layer/cmp_eq.vhd"
xvhdl $VHD_DIR"/Layer/generic_mux_1bit.vhd"
xvhdl $VHD_DIR"/Layer/generic_or.vhd"
xvhdl $DEBUG_DIR"/Layer/debug_layer.vhd"
xvhdl $VHD_DIR"/Layer/layer_cu.vhd"
xvhdl $DEBUG_DIR"/Layer/debug_layer_datapath.vhd"
xvhdl $VHD_DIR"/Layer/mux2to1.vhd"
xvhdl $VHD_DIR"/Layer/reg.vhd"

# Synapses
xvhdl $VHD_DIR"/Synapse/weights_bram.vhd"

# Shared components
xvhdl $VHD_DIR"/Shared/decoder.vhd"
xvhdl $VHD_DIR"/Shared/mux2to1_std_logic.vhd"
xvhdl $VHD_DIR"/Shared/reg_signed.vhd"
xvhdl $VHD_DIR"/Shared/cnt.vhd"

# Testbench
xvhdl $VHD_DIR"/IO/load_file.vhd"

xvhdl $TB_DIR"/debug_layer_tb.vhd"
