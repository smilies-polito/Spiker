#!/usr/bin/bash

VHD_DIR="../../Vhd/Vhd"
TB_DIR="../../Tb/Vhd"

# Neuron
xvhdl $VHD_DIR"/Neuron/add_sub.vhd"
xvhdl $VHD_DIR"/Neuron/cmp_gt.vhd"
xvhdl $VHD_DIR"/Neuron/mux2to1_signed.vhd"
xvhdl $VHD_DIR"/Neuron/mux4to1_signed.vhd"
xvhdl $VHD_DIR"/Neuron/neuron.vhd"
xvhdl $VHD_DIR"/Neuron/neuron_cu.vhd"
xvhdl $VHD_DIR"/Neuron/neuron_datapath.vhd"
xvhdl $VHD_DIR"/Neuron/reg_signed_sync_rst.vhd"
xvhdl $VHD_DIR"/Neuron/shifter.vhd"

# Bare neurons
xvhdl $VHD_DIR"/BareNeurons/bare_neurons.vhd"
xvhdl $VHD_DIR"/BareNeurons/generic_and.vhd"

# Complete neurons
xvhdl $VHD_DIR"/CompleteNeurons/complete_neurons.vhd"
xvhdl $VHD_DIR"/CompleteNeurons/double_and_mask_n.vhd"
xvhdl $VHD_DIR"/CompleteNeurons/simple_and_mask.vhd"

# Layer
xvhdl $VHD_DIR"/Layer/anticipate_bits.vhd"
xvhdl $VHD_DIR"/Layer/bit_selection.vhd"
xvhdl $VHD_DIR"/Layer/cmp_eq.vhd"
xvhdl $VHD_DIR"/Layer/generic_mux_1bit.vhd"
xvhdl $VHD_DIR"/Layer/generic_or.vhd"
xvhdl $VHD_DIR"/Layer/layer.vhd"
xvhdl $VHD_DIR"/Layer/layer_cu.vhd"
xvhdl $VHD_DIR"/Layer/layer_datapath.vhd"
xvhdl $VHD_DIR"/Layer/mux2to1.vhd"
xvhdl $VHD_DIR"/Layer/reg.vhd"

# Shared components
xvhdl $VHD_DIR"/Shared/decoder.vhd"
xvhdl $VHD_DIR"/Shared/mux2to1_std_logic.vhd"
xvhdl $VHD_DIR"/Shared/reg_signed.vhd"
xvhdl $VHD_DIR"/Shared/cnt.vhd"

# Synapses
xvhdl $VHD_DIR"/Synapse/weights_bram.vhd"

# Input interface
xvhdl $VHD_DIR"/InputInterface/input_interface.vhd"
xvhdl $VHD_DIR"/InputInterface/ff.vhd"
xvhdl $VHD_DIR"/InputInterface/lfsr.vhd"
xvhdl $VHD_DIR"/InputInterface/shift_register_unsigned.vhd"
xvhdl $VHD_DIR"/InputInterface/unsigned_cmp_gt.vhd"

# Output interface
xvhdl $VHD_DIR"/OutputInterface/out_interface.vhd"
xvhdl $VHD_DIR"/OutputInterface/out_interface_datapath.vhd"
xvhdl $VHD_DIR"/OutputInterface/out_interface_cu.vhd"

# Load file
xvhdl $VHD_DIR"/IO/load_file.vhd"

# Spiker
xvhdl $VHD_DIR"/spiker.vhd"
xvhdl $TB_DIR"/spiker_tb.vhd"

# # Input buffer
# xvhdl $VHD_DIR"/InputBuffer/input_buffer.vhd"
# xvhdl $VHD_DIR"/InputBuffer/reg_unsigned.vhd"
# 
# # Output selector
# xvhdl $VHD_DIR"/OutputSelector/generic_mux.vhd"
# 
# # Complete accelerator
# xvhdl $VHD_DIR"/CompleteAccelerator/complete_accelerator.vhd"
# 
# # Testbench
# xvhdl $TB_DIR"/CompleteAccelerator/driver.vhd"
# xvhdl $TB_DIR"/CompleteAccelerator/complete_accelerator_tb.vhd"
