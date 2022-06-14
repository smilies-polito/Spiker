#!/usr/bin/bash

ROOT_DIR=`pwd`
VHDL_SIM_DIR="../"
VHD_DIR="../vhd/vhd"
TB_DIR="../tb/vhd"
TOP_ENTITY="input_interface_tb"
COMMANDS_TCL=$ROOT_DIR"/commands.tcl"

cd $VHDL_SIM_DIR

xvhdl $VHD_DIR"/shared/mux2to1_std_logic.vhd"

xvhdl $VHD_DIR"/input_interface/ff.vhd"
xvhdl $VHD_DIR"/input_interface/shift_register_unsigned.vhd"
xvhdl $VHD_DIR"/input_interface/unsigned_cmp_gt.vhd"
xvhdl $VHD_DIR"/input_interface/lfsr.vhd"
xvhdl $VHD_DIR"/input_interface/input_interface.vhd"

xvhdl $TB_DIR"/input_interface/input_interface_tb.vhd"

xelab $TOP_ENTITY

xsim -t $COMMANDS_TCL $TOP_ENTITY

cd $ROOT_DIR
