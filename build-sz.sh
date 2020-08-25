#!/bin/bash

# jw-v2.1.7.0
set -e

module load gcc/6.1.0
module load gsl/2.5

INSTALL_PATH=/global/cscratch1/sd/jw447/local_build/SZ/install
CC=gcc

#PREFIX_PATH=/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq
#./configure --prefix=$INSTALL_PATH --enable-openmp --enable-gsl --with-gsl-prefix=$PREFIX_PATH && make -j8 && make install
./configure --prefix=$INSTALL_PATH --enable-openmp && make -j8 && make install
