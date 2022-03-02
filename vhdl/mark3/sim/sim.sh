#!/usr/bin/bash

VHD_DIR="../vhd"
TB_DIR="../tb"

TOP_ENTITY="weights_bram_tb"

xvhdl $VHD_DIR"/shared/decoder.vhd"
xvhdl $VHD_DIR"/synapse/weights_bram.vhd"
xvhdl $TB_DIR"/synapse/weights_bram_tb.vhd"

xelab $TOP_ENTITY

xsim -t commands.tcl $TOP_ENTITY
