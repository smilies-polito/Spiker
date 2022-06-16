#!/usr/bin/bash

TOP_ENTITY="spiker_tb"

xelab $TOP_ENTITY

xsim -t commands.tcl $TOP_ENTITY
