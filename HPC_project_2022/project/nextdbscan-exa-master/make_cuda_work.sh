#! /bin/sh
MAX_GCC_VERSION=8
ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/local/cuda/bin/gcc
ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/local/cuda/bin/g++