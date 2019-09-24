#!/usr/bin/env python

import json
import subprocess
import re
import os

with open('compile_commands.json', 'r') as f:
    compile_commands = json.load(f)

# # remove repeating commands for the same source files in the same directory
# seen = []
# for comp_unit in compile_commands:
#     notseen = True
#     for s in seen:
#         if s['file'] == comp_unit['file'] and s['directory'] == comp_unit['directory']:
#             notseen = False
#     if notseen:
#         seen.append(comp_unit) 
# compile_commands = seen



# # extract list of local (non-system) headers that are used in each compilation unit 
# for comp_unit in compile_commands:
#     print("%s %s" % (comp_unit['file'], comp_unit['directory']))
#     command = comp_unit['arguments']
# 
#     # remove output file declarations
#     while '-o' in command:
#         output_index = command.index('-o')
#         # delete option '-o' and the specified file
#         del command[output_index]
#         del command[output_index]
# 
#     # remove object files '*.o'
#     command = filter(lambda s: s[-2:] != '.o', command)
# 
#     command.append('-M') # compiler will provide us the list of included headers
# 
#     # run command
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=comp_unit['directory'])
#     # wait for process finish and grab its standard output
#     output, _ = process.communicate()
#     output = re.split('[\s\\\\]+', output) # remove spaces and slashes
#     output = filter(None, output) # remove empty strings in the beginning/end of list
#     output = filter(lambda s: s[0] != '/', output) # remove system headers (they have leading '/')
#     output = filter(lambda s: s[-1] != ':', output) # remove dependency (they have ':' at the end)
# 
#     # now all that left in output is actual source file and all headers
#     print(output)

for comp_unit in compile_commands:
    
    ### run converter for each file

    converter = "$HOME/code/build-gpumpi-Desktop-Debug/_deps/llvm-build/bin/converter"
    converter = os.path.expandvars(converter)
    #print(comp_unit['directory'])
    #args = comp_unit['arguments'][1:]
    #command = [converter] + output + ["--"] + args
    command = [converter, comp_unit['file'], "-p", "compile_commands.json"]
    print(" ".join(command))
    process = subprocess.Popen(command, cwd=comp_unit['directory'])
    process.wait()
    if process.returncode != 0:
        raise Exception("failed")
    

    find . -name '*\.cuh?'
