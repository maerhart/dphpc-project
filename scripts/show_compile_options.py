#!/usr/bin/env python

import json
import subprocess
import re
import os
import argparse

parser = argparse.ArgumentParser(description='find includes for given source in compile_commands.json')
parser.add_argument("input_file", type=argparse.FileType('rb'), help="path to the source file")

args = parser.parse_args()

args.input_file

input_path = os.path.realpath(args.input_file.name)

with open('compile_commands.json', 'r') as f:
    compile_commands = json.load(f)

compile_options = []

for cu in compile_commands:
    path = os.path.realpath(cu['directory'] + '/' + cu['file'])
    if path == input_path:
        compile_options += cu['arguments']

for opt in compile_options:
    print(opt)
