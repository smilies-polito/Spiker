#!/usr/bin/bash

ROOT_DIR=`pwd`

VHDL_SIM_DIR="../../../../Vhdl/Hierarchical/Sim/VivadoScripts"

echo "VHDL simulation"
cd $VHDL_SIM_DIR
./sim.sh > /dev/null
cd $ROOT_DIR
