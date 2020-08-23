#!/bin/bash

scriptdir=$(dirname "$0")

"$scriptdir/../../scripts/convert_and_build.sh"

binary=$(find "$scriptdir/gpumpi_build" -name 'target_*_cpi_c')

"$binary" ---gpumpi -g 4 -b 1
