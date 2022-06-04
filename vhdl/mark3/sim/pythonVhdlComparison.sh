#!/usr/bin/bash

ROOT_DIR=`pwd`

BRAM_INIT_DIR="bramInit"
PYTHON_SIM_DIR="../../../python/custom/mark4_finitePrecision"
VHDL_SIM_DIR="vivadoScripts"
INPUT_GEN_DIR="inputGen"
IMAGE_INDEX=30

echo "Generate input spikes"
cd $INPUT_GEN_DIR
python imgToSpikes.py $IMAGE_INDEX
cd $ROOT_DIR

echo "Python simulation"
cd $PYTHON_SIM_DIR
python mnistTest.py $IMAGE_INDEX
cd $ROOT_DIR

echo "Format BRAM and threshold initialization files"
cd $BRAM_INIT_DIR
python bramInit.py
cd $ROOT_DIR

echo "VHDL simulation"
cd $VHDL_SIM_DIR
./debug_compile.sh
./debug_sim.sh
cd $ROOT_DIR
