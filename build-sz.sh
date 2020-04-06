#!/bin/bash

set -e

module load gcc/6.1.0

INSTALL_PATH=/global/cscratch1/sd/jw447/local_build/SZ/install

CC=gcc
./configure --prefix=$INSTALL_PATH --enable-openmp && make -j8 && make install
