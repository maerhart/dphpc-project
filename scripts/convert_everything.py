#!/usr/bin/env python

import json
import subprocess
import re
import os

with open('compile_commands.json', 'r') as f:
    compile_commands = json.load(f)

for comp_unit in compile_commands:
    
    ### run converter for each file

    converter = "$HOME/code/build-gpumpi-Desktop-Debug/_deps/llvm-build/bin/converter"
    converter = os.path.expandvars(converter)
    command = [converter, comp_unit['file'], "-p", "compile_commands.json"]
    print(" ".join(command))
    process = subprocess.Popen(command, cwd=comp_unit['directory'])
    process.wait()
    if process.returncode != 0:
        raise Exception("failed")
    

