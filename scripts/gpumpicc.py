#!/usr/bin/env python

import sys
import os
import subprocess
import re
import shutil
from typing import List, Tuple

source_extensions = (".c", ".cpp", ".cxx")

def detect_sources_and_args(comilation_params: List[str]) -> Tuple[List[str], List[str]]:
    compilation_sources = []
    compilation_args = []

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

    args_iter = iter(compilation_args)
    for arg in args_iter:
        if arg in valid_prefixes:
            prefix = arg
            path = next(args_iter)
            clean_args.append(prefix + path)
        elif arg.startswith(valid_prefixes):
            clean_args.append(arg)
        elif arg in single_word_args:
            clean_args.append(arg)
        elif arg in double_word_args:
            clean_args.append(arg)
            clean_args.append(next(args_iter))
        elif arg.endswith(".o"):
            clean_args.append(arg)

    return clean_args


def get_header_depenency_lists(comilation_params: List[str]) -> List[Tuple[str, List[str]]]:
    # remove -o option capture result from standard output
    fixed_params = []
    comp_params_iter = iter(compilation_params)
    for param in comp_params_iter:
        if param == '-o':
            # skip parameter following '-o'
            next(comp_params_iter) 
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


def convert_source_name(name: str) -> str:
    """ change suffix of source file to 'cu'. For example: foo.c -> foo.cu """

    assert name.lower().endswith(source_extensions)
    
    name_parts = name.split(".")
    name_parts[-1] = 'cu'

    return ".".join(name_parts)


if __name__ == '__main__':
    
    scripts_dir = os.path.abspath(os.path.dirname(__file__))
    converter_path = scripts_dir + "/../source_converter/converter"

    compilation_params: List[str] = sys.argv[1:]
    compilation_sources, compilation_args = detect_sources_and_args(compilation_params)

    mpi_compile_flags, mpi_link_flags = get_mpi_flags()

    # TODO: ugly: location of static libraries is controlled by cmake and can be changed in newer cmake versions
    gpu_mpi_libs: List[str] = [ 
        scripts_dir + "/../gpu_libs/gpu_mpi/libgpu_mpi.a",
        scripts_dir + "/../gpu_libs/cuda_btl/libcuda_btl.a",
        scripts_dir + "/../gpu_libs/gpu_libc/libgpu_libc.a",
        scripts_dir + "/../gpu_libs/gpu_main/libgpu_main.a",
        scripts_dir + "/../gpu_libs/gpu_main/liblibc_processor.a",
        scripts_dir + "/../common/libcommon.a",
    ]

    # WARNING: will not work if source dir is moved or deleted
    gpu_mpi_headers: List[str] = [ 
        "@CMAKE_SOURCE_DIR@/gpu_libs/gpu_mpi",
        "@CMAKE_SOURCE_DIR@/gpu_libs/gpu_libc",
        "@CMAKE_SOURCE_DIR@/gpu_libs/gpu_main",
    ]

    gpu_mpi_include_args: List[str] = ['-I' + h for h in gpu_mpi_headers]

    # if there are no compilation_sources, then probably compiler wrapper is used for linking only, so we will not run converter on it
    if compilation_sources:

        tu_header_deps = get_header_depenency_lists(compilation_params)

        # Before running converter, we manually create copies of files with ".cu" or ".cuh" suffix, because
        # when converter doesn't make any modifications to the file, it just skips it.
        for tu_dep in tu_header_deps:
            source_file, headers = tu_dep

            new_source = convert_source_name(source_file)
            shutil.copyfile(source_file, new_source)

            for header in headers:
                new_header = header + '.cuh'
                shutil.copyfile(header, new_header)

        # run code converter
        command = [converter_path, *compilation_sources, "--", *compilation_args, *mpi_compile_flags]
        #print("command:", " ".join(command))
        subprocess.run(command, check=True)
    
        for tu_dep in tu_header_deps:
            source_file, headers = tu_dep

            for file_name in (source_file, *headers):
                # 1. for source files replace filename extension by '.cu'
                # it is required because object file will have the same name with .cu replaced by .o
                # so existing builds will not break due to name changes
                # 2. for headers append '.cuh' to it without removing anything
                # it is required to simplify providing our own headers instead of standard ones
                if file_name == source_file:
                    cu_file_name = convert_source_name(file_name)
                else:
                    cu_file_name = file_name + '.cuh'

                # add .cuh suffix to #include directives
                with open(cu_file_name, 'r') as in_file:
                    text = re.sub('#include ([<"])(.*)([">])', '#include \g<1>\g<2>.cuh\g<3>', in_file.read())
                    text = re.sub('\.cuh\.cuh', '.cuh', text) # fixes double addition of '.cuh' suffix if we already processed this file
                with open(cu_file_name, 'w') as out_file:
                    out_file.write(text)

    cuda_sources = [convert_source_name(s) for s in compilation_sources]
    cuda_args = cuda_compatible_compilation_args(compilation_args)

    gpu_arch_flags = '@CMAKE_CUDA_FLAGS@'.split() # pass gpu arch

    # pass debug compile flags if they present
    if "@CMAKE_BUILD_TYPE@" == "Debug":
        gpu_arch_flags += '@CMAKE_CUDA_FLAGS_DEBUG@'.split()

    #print('CMAKE_CUDA_FLAGS', gpu_arch_flags)

    command = ['nvcc', '-rdc=true', *gpu_arch_flags, *cuda_sources, *cuda_args, *gpu_mpi_libs, *gpu_mpi_include_args]
    #print("compile command:", " ".join(command))
    subprocess.run(command, check=True)
    
    
