#!/usr/bin/env python

import json
import subprocess
import os
import re
import shutil
import textwrap

def is_inside(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    return directory == os.path.commonpath([directory, path])

def escape_name(name):
    return name.replace('/', '_').replace('.', '_')


header_pattern = re.compile('#include [<"](.*)[">]')

def find_headers(path_to_file, include_dirs):
    """ Looks for header inside include_dirs and header local directory.
    Returns list of detected headers.
    """

    file_dir = os.path.dirname(os.path.realpath(__file__))

    # we should search relative headers in current dir
    all_include_dirs = [file_dir] + include_dirs

    detected_headers = []
    with open(path_to_file, 'r') as f:
        for match in header_pattern.finditer(f.read()):
            header_name = match.group(1)

            # search relative headers in include dirs and detect their absolute location
            absolute_header = None
            if not os.path.isabs(header_name):
                for include_dir in all_include_dirs:
                    header_candidate = os.path.join(include_dir, header_name)
                    if os.path.exists(header_candidate):
                        absolute_header = os.path.realpath(header_candidate)
                        break

            # if nothing is find, then it is system header that should be skipped
            if absolute_header is None:
                continue

            detected_headers.append(absolute_header)

    all_headers = detected_headers
    
    # for each detected header we need to look for other includes recurrently
    for header in detected_headers:
        all_headers += find_headers(header, include_dirs)

    # return each header once
    all_headers = list(set(all_headers))

    return all_headers

def get_includes(absolute_path, compile_commands):
    for entry in compile_commands:
        if entry['absolute_source_path'] == absolute_path:
            return entry['project_include_dirs']
    raise Exception(f'Entry {absolute_path} not found in compilation database')

def get_definitions(absolute_path, compile_commands):
    for entry in compile_commands:
        if entry['absolute_source_path'] == absolute_path:
            return entry['definitions']
    raise Exception(f'Entry {absolute_path} not found in compilation database')

def run_build():
    os.makedirs('./gpumpi_build')
    process = subprocess.Popen("cmake ..".split(), cwd='./gpumpi_build')
    process.wait()
    process = subprocess.Popen("cmake --build ./gpumpi_build".split())
    process.wait()

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.getcwd()

    def run_cmd(command, directory):
        process = subprocess.Popen(command, cwd=directory)
        process.wait()
        if process.returncode != 0:
            raise Exception("failed")

    with open('compile_commands.json', 'r') as f:
        compile_commands = json.load(f)


    all_sources = []
    all_headers = []

    for entry in compile_commands:
        # detect include directories inside project dir for each target
        entry['project_include_dirs'] = []
        for arg in entry['arguments']:
            # detect only include directories
            if not arg.startswith('-I'):
                continue

            include_dir = arg[2:]

            # make all paths absolute
            if not os.path.isabs(include_dir):
                include_dir = os.path.realpath(os.path.join(entry['directory'], include_dir))

            # skip includes outside of project directory
            if not is_inside(include_dir, project_dir):
                continue 

            entry['project_include_dirs'].append(include_dir)

        # detect definitions
        entry['definitions'] = []
        for arg in entry['arguments']:
            if arg.startswith('-D'):
                entry['definitions'].append(arg[2:])


        # detect absolute path to source files
        entry['absolute_source_path'] = os.path.realpath(os.path.join(entry['directory'], entry['file']))

        # detect list of headers inside project directory that are used from source files 
        all_headers += find_headers(entry['absolute_source_path'], entry['project_include_dirs'])
        all_sources += [entry['absolute_source_path']]

    # mention sources and headers only once
    all_sources = list(set(all_sources))
    all_headers = list(set(all_headers))

    # create ".cuh" for very simple headers, because libtooling skips them
    for header in all_headers:
        expected_name = header + '.cuh'
        if not os.path.exists(expected_name):
            shutil.copyfile(header, expected_name)

    # add to each include in '.cu' or '.cuh' file additional '.cuh' suffix
    for file_name in (*all_sources, *all_headers):

        if file_name in all_sources:
            cu_file_name = file_name + '.cu'
        else:
            cu_file_name = file_name + '.cuh'

        with open(cu_file_name, 'r') as in_file: 
            text = re.sub('#include ([<"])(.*)([">])', '#include \g<1>\g<2>.cuh\g<3>', in_file.read())
            text = re.sub('\.cuh\.cuh', '.cuh', text) # fixes double modification
        with open(cu_file_name, 'w') as out_file:
            out_file.write(text)


    # for each source file detect if __gpu_main present. If yes, it will define executable,
    # otherwise, it will define library.
    executables = []
    libraries = []
    for file_name in all_sources:
        cu_file_name = file_name + '.cu'
        with open(cu_file_name, 'r') as f:
            if '__gpu_main' in f.read():
                executables.append(file_name)
            else:
                libraries.append(file_name)

    cmakelists = textwrap.dedent(f"""
        cmake_minimum_required(VERSION 3.12)
        project(examples LANGUAGES C CXX CUDA)

        set(CMAKE_CUDA_FLAGS \
            "${{CMAKE_CUDA_FLAGS}} \
            -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_70,code=sm_70")

        include({script_dir}/../gpu_libs-exports.cmake)

        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    """)

    for f in all_sources:
        includes = get_includes(f, compile_commands)
        defines = get_definitions(f, compile_commands)
        escaped_name = escape_name(f)
        target_type = 'executable' if f in executables else 'library'

        cmakelists += textwrap.dedent(f"""
            add_{target_type}(target_{escaped_name} {f}.cu)
            target_link_libraries(target_{escaped_name} PRIVATE gpu_libs)
        """)

        if includes:
            includes_str = " ".join(includes)
            cmakelists += textwrap.dedent(f"""
                target_include_directories(target_{escaped_name} PRIVATE {includes_str})
            """)

    for lib in libraries:
        for exe in executables:
            escaped_lib_name = escape_name(lib)
            escaped_exe_name = escape_name(exe)
            cmakelists += textwrap.dedent(f"""
                target_link_libraries(target_{escaped_exe_name} PRIVATE target_{escaped_lib_name})
            """)

    with open('CMakeLists.txt', 'w') as f:
        f.write(cmakelists)

    run_build()

