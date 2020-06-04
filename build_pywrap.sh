#!/bin/bash

set -e

module load cmake/3.14.4
#module load gcc/6.1.0

INSTALL_PATH=/global/cscratch1/sd/jw447/local_build/SZ/install

#CC=gcc

cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_PYTHON_WRAPPER=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC ..
cmake --build .
cmake --install
