#!/bin/bash

set -e

module load gcc/6.1.0
module load gsl/2.5

INSTALL_PATH=/global/cscratch1/sd/jw447/local_build/SZ/install
PREFIX_PATH=/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq
CC=gcc
./configure --prefix=$INSTALL_PATH --enable-openmp --enable-gsl --with-gsl-prefix=$PREFIX_PATH && make -j8 && make install
