#!/bin/bash

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

"$scriptdir/../../scripts/convert_and_build.sh"

binary=$(find "$scriptdir/gpumpi_build" -name 'target_*_cpi_c')

"$binary" ---gpumpi -g 4 -b 1
