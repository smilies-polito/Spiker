#!/usr/bin/bash

ROOT_DIR=$(pwd)

VHDL_DIR="../../../Vhdl"
PYTHON_SIM_DIR=$ROOT_DIR

BRAM_INIT_DIR="$PYTHON_SIM_DIR/BramInit"
PYTHON_MODEL="$PYTHON_SIM_DIR/PythonModel"
COUNTERS_CONVERTION_DIR="$PYTHON_SIM_DIR/VhdlCountersConvertion"
VHDL_SIM_DIR="$VHDL_DIR/Hierarchical/Sim/VivadoScripts"
INPUT_GEN_DIR="$PYTHON_SIM_DIR/SpikesGen"
IMAGE_INDEX=30

echo "Generate input spikes"
cd $INPUT_GEN_DIR
python imgToSpikes.py $IMAGE_INDEX
cd $ROOT_DIR

echo "Python simulation"
cd $PYTHON_MODEL
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

echo "VHDL counters translation"
cd $COUNTERS_CONVERTION_DIR
python countersConvertion.py
cd $ROOT_DIR
