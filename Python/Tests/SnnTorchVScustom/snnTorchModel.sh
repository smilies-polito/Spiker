#!/usr/bin/bash

source files.sh

cd $snntorch_dir
python mnistTest.py
cd $current_dir
