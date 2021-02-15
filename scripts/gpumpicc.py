#!/usr/bin/env python

import sys
import os
import subprocess
import re
import shutil
from typing import List, Tuple

def detect_sources_and_args(comilation_params: List[str]) -> Tuple[List[str], List[str]]:
    compilation_sources = []
    compilation_args = []

    source_extensions = (".c", ".cpp", ".cxx")

    for arg in compilation_params:
        if arg.lower().endswith(source_extensions):
            compilation_sources.append(arg)
        else:
            compilation_args.append(arg)

    return compilation_sources, compilation_args
    

def get_mpi_flags() -> Tuple[List[str], List[str]]:
    # TODO this is openmpi and C specific, for other mpi wrappers add language pairs add corresponding checks 
    cp: subprocess.CompletedProcess = subprocess.run(["mpicc", "--showme:compile"], check=True, capture_output=True)
    mpi_compile_flags = cp.stdout.decode("utf-8").strip().split(" ")

    cp: subprocess.CompletedProcess = subprocess.run(["mpicc", "--showme:link"], check=True, capture_output=True)
    mpi_link_flags = cp.stdout.decode("utf-8").strip().split(" ")

    return mpi_compile_flags, mpi_link_flags


def cuda_compatible_compilation_args(compilation_args: List[str]) -> List[str]:
    """ Removes compilation args unknown for nvcc compiler or irrelevant for gpu_mpi project """

    clean_args = []

    valid_prefixes = ("-I", "-D")

    single_word_args = ('-c')
    double_word_args = ('-o')

    for arg in compilation_args:
        if arg in valid_prefixes:
            prefix = arg
            path = next(compilation_args)
            clean_args.append(prefix + path)
        elif arg.startswith(valid_prefixes):
            clean_args.append(arg)
        elif arg in single_word_args:
            clean_args.append(arg)
        elif arg in double_word_args:
            clean_args.append(arg)
            clean_args.append(next(compilation_args))
        elif arg.endswith(".o"):
            clean_args.append(arg)

    return clean_args


def get_header_depenency_lists(comilation_params: List[str]) -> List[Tuple[str, List[str]]]:
    # remove -o option capture result from standard output
    fixed_params = []
    for param in compilation_params:
        if param == '-o':
            # skip parameter following '-o'
            next(compilation_params) 
            continue

        fixed_params.append(param)

    cp: subprocess.CompletedProcess = subprocess.run(['mpicc', *fixed_params, '-M'], check=True, capture_output=True)
    
    header_deps = cp.stdout.decode("utf-8").strip().replace('\\\n', ' ')

    # split into different translation units
    unparsed_tu_deps = header_deps.split('\n')

    project_dir = os.path.realpath(os.environ['GPU_MPI_PROJECT'])

    tu_header_deps = []
    for dep in unparsed_tu_deps:
        header_deps_list = dep.split()

        source_file = header_deps_list[1]
        # remove name of target and name of source file
        headers = header_deps_list[2:]

        # remove headers that are not in project dir
        project_headers = [h for h in headers if os.path.realpath(h).startswith(project_dir)]

        tu_header_deps.append((source_file, project_headers))

    return tu_header_deps


if __name__ == '__main__':
    
    scripts_dir = os.path.abspath(os.path.dirname(__file__))
    converter_path = scripts_dir + "/../source_converter/converter"

    compilation_params: List[str] = sys.argv[1:]
    compilation_sources, compilation_args = detect_sources_and_args(compilation_params)

    mpi_compile_flags, mpi_link_flags = get_mpi_flags()

    # if there are no compilation_sources, then probably compiler wrapper is used for linking only, so we will not run converter on it
    if compilation_sources:

        # run code converter
        command = [converter_path, *compilation_sources, "--", *compilation_args, *mpi_compile_flags]
        #print("command:", " ".join(command))
        subprocess.run(command, check=True)
    
        # add .cuh suffix to #include directives
        tu_header_deps = get_header_depenency_lists(compilation_params)

        for tu_dep in tu_header_deps:
            source_file, headers = tu_dep

            # some headers may not exist, because converter didn't make any modifications to them
            # we have to make copy of such headers with '.cuh' suffix for consistency
            for header in headers:
                expected_name = header + '.cuh'
                if not os.path.exists(expected_name):
                    shutil.copyfile(header, expected_name)

            for file_name in (source_file, *headers):
                if file_name == source_file:
                    cu_file_name = file_name + '.cu'
                else:
                    cu_file_name = file_name + '.cuh'

                with open(cu_file_name, 'r') as in_file:
                    text = re.sub('#include ([<"])(.*)([">])', '#include \g<1>\g<2>.cuh\g<3>', in_file.read())
                    text = re.sub('\.cuh\.cuh', '.cuh', text) # fixes double addition of '.cuh' suffix if we already processed this file
                with open(cu_file_name, 'w') as out_file:
                    out_file.write(text)

    cuda_sources = [s + ".cu" for s in compilation_sources]
    cuda_args = cuda_compatible_compilation_args(compilation_args)

    command = ['nvcc', *cuda_sources, *cuda_args]
    print("command:", " ".join(command))
    subprocess.run(command, check=True)
    
    
