#!/bin/bash

set -e

set -x

INSTALL_PATH=/home/jon/local_build/SZ/install

./configure --prefix=$INSTALL_PATH && make && make install

