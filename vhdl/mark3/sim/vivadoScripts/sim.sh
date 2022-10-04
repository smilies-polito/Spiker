#!/usr/bin/bash

TOP_ENTITY="complete_accelerator_tb"

xelab $TOP_ENTITY

xsim -t commands.tcl $TOP_ENTITY
