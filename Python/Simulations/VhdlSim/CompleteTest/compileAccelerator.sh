#!/usr/bin/bash

ROOT_DIR=`pwd`

VHDL_SIM_DIR="../../../../Vhdl/Hierarchical/Sim/VivadoScripts"

echo "VHDL compile"
cd $VHDL_SIM_DIR
./compile.sh #> /dev/null
cd $ROOT_DIR
