#!/bin/bash

set -x
set -e

export SCRIPTDIR=$(dirname "$0")

make clean && bear make
"$SCRIPTDIR"/convert_everything.py
"$SCRIPTDIR"/build_on_gpu.sh

