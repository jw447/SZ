#!/bin/bash

set -e

module load gcc/6.1.0
CC=gcc

INSTALL_PATH=/global/cscratch1/sd/jw447/local_build/SZ/install

cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -D-DBUILD_PYTHON_WRAPPER=ON -DBUILD_SHARED_LIBS=ON ..
cmake --build
cmake --install
