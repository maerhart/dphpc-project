#!/bin/bash

SHELLFILE_PATH=$(dirname "$0")
GPU_MPI_PROJECT=$SHELLFILE_PATH/../.. $SHELLFILE_PATH/../../build/source_converter/converter \
$1 --write_to_stdout $2 -- \
-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent \
-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include \
-I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread
