#!/bin/bash
GLIBC_PATH=/home/ppc17public/gnu/glibc-2.23/build
$GLIBC_PATH/lib/ld-2.23.so --library-path $GLIBC_PATH/lib:$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/lib64:/usr/lib:/usr/lib64:/lib:/lib64 /home/ppc17public/Python-2.7.11/build/bin/python2.7 $@
