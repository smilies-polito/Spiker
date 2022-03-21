#!/usr/bin/bash

VHD_DIR="../vhd"
TB_DIR="../tb"

TOP_ENTITY="string_man"

xvhdl $TB_DIR"/stringManipulation.vhd"

xelab $TOP_ENTITY

xsim -t commands.tcl $TOP_ENTITY
