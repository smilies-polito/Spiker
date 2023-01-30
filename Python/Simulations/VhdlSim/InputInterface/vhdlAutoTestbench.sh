#!/usr/bin/bash

ROOT_DIR=`pwd`
VHDL_SIM_DIR="../../../../Vhdl/Hierarchical/Sim"
VHD_DIR="../Vhd/Vhd"
TB_DIR="../Tb/Vhd"
TOP_ENTITY="input_interface_tb"
COMMANDS_TCL="InputInterface/commands.tcl"

cd $VHDL_SIM_DIR

xvhdl $VHD_DIR"/Shared/mux2to1_std_logic.vhd"

xvhdl $VHD_DIR"/InputInterface/ff.vhd"
xvhdl $VHD_DIR"/InputInterface/shift_register_unsigned.vhd"
xvhdl $VHD_DIR"/InputInterface/unsigned_cmp_gt.vhd"
xvhdl $VHD_DIR"/InputInterface/lfsr.vhd"
xvhdl $VHD_DIR"/InputInterface/input_interface.vhd"

xvhdl $TB_DIR"/InputInterface/input_interface_tb.vhd"

xelab $TOP_ENTITY

xsim -t $COMMANDS_TCL $TOP_ENTITY

cd $ROOT_DIR
