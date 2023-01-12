#!/usr/bin/bash

TOP_ENTITY="debug_layer_tb"

xelab $TOP_ENTITY

xsim -t commands.tcl $TOP_ENTITY
