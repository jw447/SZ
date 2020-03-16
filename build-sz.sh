#!/bin/bash

set -e

INSTALL_PATH=/global/cscratch1/sd/jw447/local_build/SZ/install
#
./configure --prefix=$INSTALL_PATH && make && make install
