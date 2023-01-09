#!/usr/bin/bash

TOP_ENTITY="complete_accelerator"

xelab $TOP_ENTITY

#xsim -t commands.tcl $TOP_ENTITY
xsim $TOP_ENTITY
