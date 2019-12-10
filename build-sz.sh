#!/bin/bash

set -e

INSTALL_PATH=/global/homes/j/jw447/local_build/SZ/install
#
./configure --prefix=$INSTALL_PATH && make && make install
