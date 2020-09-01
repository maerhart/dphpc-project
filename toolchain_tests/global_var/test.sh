#!/bin/bash

scriptdir=$(dirname "$0")

cd "$scriptdir"

"$scriptdir/../../scripts/convert_and_build.sh"

binary=$(find "$scriptdir/gpumpi_build" -name 'target_*_global_var_c')

"$binary" ---gpumpi -g 4 -b 1
