#!/usr/bin/bash

VHD_DIR="../vhd"
TB_DIR="../tb"

TOP_ENTITY="load_file_tb"

xvhdl $VHD_DIR"/io/load_file.vhd"
xvhdl $TB_DIR"/io/load_file_tb.vhd"

xelab $TOP_ENTITY

xsim -t commands.tcl $TOP_ENTITY
