#!/usr/bin/env python

import json
import subprocess
import re
import os

with open('compile_commands.json', 'r') as f:
    compile_commands = json.load(f)

for comp_unit in compile_commands:
    ### run converter for each file

    file = comp_unit['file']
    directory = comp_unit['directory']

    # skip unsupported formats
    extensions = (".c", ".cpp", ".cxx")
    if not file.lower().endswith(extensions):
        continue

    converter = "$SCRIPTDIR/../source_converter/converter"

    converter = os.path.expandvars(converter)
    comp_db = os.path.abspath("compile_commands.json")
    command = [converter, file, "-p", comp_db]
    print(" ".join(command))
    print(directory)
    process = subprocess.Popen(command, cwd=directory)
    process.wait()
    if process.returncode != 0:
        raise Exception("failed")
    


