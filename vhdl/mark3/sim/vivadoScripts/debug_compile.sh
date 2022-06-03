#!/usr/bin/bash

VHD_DIR="../../vhd/vhd"
DEBUG_DIR="../../vhd/debug"
TB_DIR="../../tb/debug/layer"

TOP_ENTITY="debug_layer_tb"

# Neuron
xvhdl $VHD_DIR"/neuron/add_sub.vhd"
xvhdl $VHD_DIR"/neuron/cmp_gt.vhd"
xvhdl $VHD_DIR"/neuron/mux2to1_signed.vhd"
xvhdl $VHD_DIR"/neuron/mux4to1_signed.vhd"
xvhdl $VHD_DIR"/neuron/neuron_cu.vhd"
xvhdl $VHD_DIR"/neuron/neuron_datapath.vhd"
xvhdl $VHD_DIR"/neuron/neuron.vhd"
xvhdl $DEBUG_DIR"/neuron/debug_neuron_datapath.vhd"
xvhdl $DEBUG_DIR"/neuron/debug_neuron.vhd"
xvhdl $VHD_DIR"/neuron/reg_signed_sync_rst.vhd"
xvhdl $VHD_DIR"/neuron/shifter.vhd"

# Bare neurons
xvhdl $DEBUG_DIR"/bare_neurons/debug_bare_neurons.vhd"
xvhdl $VHD_DIR"/bare_neurons/generic_and.vhd"

# Complete neurons
xvhdl $DEBUG_DIR"/complete_neurons/debug_complete_neurons.vhd"
xvhdl $VHD_DIR"/complete_neurons/double_and_mask_n.vhd"
xvhdl $VHD_DIR"/complete_neurons/simple_and_mask.vhd"

# Layer
xvhdl $VHD_DIR"/layer/anticipate_bits.vhd"
xvhdl $VHD_DIR"/layer/bit_selection.vhd"
xvhdl $VHD_DIR"/layer/cmp_eq.vhd"
xvhdl $VHD_DIR"/layer/cnt.vhd"
xvhdl $VHD_DIR"/layer/generic_mux_1bit.vhd"
xvhdl $VHD_DIR"/layer/generic_or.vhd"
xvhdl $DEBUG_DIR"/layer/debug_layer.vhd"
xvhdl $VHD_DIR"/layer/layer_cu.vhd"
xvhdl $DEBUG_DIR"/layer/debug_layer_datapath.vhd"
xvhdl $VHD_DIR"/layer/mux2to1.vhd"
xvhdl $VHD_DIR"/layer/reg.vhd"

# Synapses
xvhdl $VHD_DIR"/synapse/weights_bram.vhd"

# Shared components
xvhdl $VHD_DIR"/shared/decoder.vhd"
xvhdl $VHD_DIR"/shared/mux2to1_std_logic.vhd"
xvhdl $VHD_DIR"/shared/reg_signed.vhd"

# Testbench
xvhdl $VHD_DIR"/io/load_file.vhd"

xvhdl $TB_DIR"/debug_layer_tb.vhd"
