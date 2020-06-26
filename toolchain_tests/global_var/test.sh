#!/bin/bash

../../scripts/convert_and_build.sh && gpumpi_build/target_bin___global_var_c_cu ---gpumpi -g 4 -b 1
