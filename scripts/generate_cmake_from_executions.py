#!/usr/bin/env python


import os
from enum import Enum, auto
from textwrap import dedent
from pathlib import Path


class TargetType(Enum):
    SOURCE = auto()
    SHARED_LIB = auto()
    STATIC_LIB = auto()
    EXECUTABLE = auto()
    UNKNOWN = auto()


def make_absolute_path(dir, path):
    return os.path.abspath(os.path.join(dir, path))


def make_unique_name(dir, path):
    abspath = make_absolute_path(dir, path)
    return "".join([x if x.isalnum() else "_" for x in abspath])


class Target:

    def __init__(self, binary_type, output, target_type, sources, cwd, cmd, include_dirs, definitions,
                 link_libraries, compile_options, objects):
        self.binary_type = binary_type
        self.output = output
        self.target_type = target_type
        self.sources = [make_absolute_path(cwd, s) for s in sources]
        self.cwd = cwd
        self.cmd = cmd
        self.include_dirs = [make_absolute_path(cwd, id) for id in include_dirs]
        self.definitions = definitions
        self.link_libraries = link_libraries
        self.compile_options = compile_options
        self.objects = [make_unique_name(self.cwd, o) for o in objects]

        self.name = make_unique_name(self.cwd, self.output)


class BinaryType(Enum):
    C_COMPILER = auto()
    CPP_COMPILER = auto()
    LINKER = auto()
    ARCHIVER = auto()
    UNKNOWN = auto()


def detect_binary_type(binary):
    if binary.endswith('gcc'):
        return BinaryType.C_COMPILER
    elif binary.endswith('g++'):
        return BinaryType.CPP_COMPILER
    elif binary.endswith('ld'):
        return BinaryType.LINKER
    elif binary.endswith('ar'):
        return BinaryType.ARCHIVER
    else:
        return BinaryType.UNKNOWN


def detect_output(binary_type, cmd):
    if binary_type is BinaryType.C_COMPILER or binary_type is BinaryType.CPP_COMPILER:
        target_type = None
        if '-shared' in cmd:
            target_type = TargetType.SHARED_LIB
        elif '-c' in cmd:
            target_type = TargetType.STATIC_LIB
        else:
            target_type = TargetType.EXECUTABLE

        for i, token in enumerate(cmd):
            if token == '-o':
                target_name = cmd[i+1]
                return target_name, target_type

        # what if '-o' option is not specified?
        raise NotImplementedError

    elif binary_type is BinaryType.LINKER:
        if '-shared' in cmd:
            target_type = TargetType.SHARED_LIB
        else:
            target_type = TargetType.STATIC_LIB

        for i, token in enumerate(cmd):
            if token == '-o':
                target_name = cmd[i+1]
                return target_name, target_type

        raise LookupError

    elif binary_type is BinaryType.ARCHIVER:
        for token in cmd:
            if token.endswith('.a'):
                return token, TargetType.STATIC_LIB
        raise LookupError
    else:
        return None, None


def detect_sources(execution):
    sources = []
    cmd = execution[1]
    for e in cmd:
        if e.endswith('.c') or e.endswith('.cpp'):
            sources.append(e)
    return sources


def detect_objects(binary_type, execution):
    cmd = execution[1]
    if binary_type is BinaryType.CPP_COMPILER or binary_type is BinaryType.C_COMPILER:
        objects = []
        for i, e in enumerate(cmd):
            if (e.endswith('.o') or e.endswith('.lo') or e.endswith('.a') or e.endswith('.so'))\
                    and cmd[i-1] != '-o':
                objects.append(e)
        return objects
    elif binary_type is BinaryType.ARCHIVER:
        objects = []
        started = False
        for i, e in enumerate(cmd):
            if started:
                objects.append(e)
            elif e.endswith('.a'):
                started = True
        return objects
    else:
        return []


def detect_link_libraries(binary_type, execution):
    if binary_type is BinaryType.CPP_COMPILER or binary_type is BinaryType.C_COMPILER:
        link_libraries = []
        cmd = execution[1]
        for i, e in enumerate(cmd):
            if e == '-l':
                link_libraries.append(execution.cmd[i+1])
            elif e.startswith('-l'):
                link_libraries.append(e[2:])
        return link_libraries
    else:
        return []


def detect_include_dirs(binary_type, execution):
    if binary_type is BinaryType.CPP_COMPILER or binary_type is BinaryType.C_COMPILER:
        include_dirs = []
        cmd = execution[1]
        for i, e in enumerate(cmd):
            if e == '-I':
                include_dirs.append(execution.cmd[i+1])
            elif e.startswith('-I'):
                include_dirs.append(e[2:])
        return include_dirs
    else:
        return []


def detect_definitions(binary_type, execution):
    if binary_type is BinaryType.CPP_COMPILER or binary_type is BinaryType.C_COMPILER:
        definitions = []
        cmd = execution[1]
        for i, e in enumerate(cmd):
            if e == '-D':
                definitions.append(execution.cmd[i+1])
            elif e.startswith('-D'):
                definitions.append(e[2:])
        return definitions
    else:
        return []


def detect_compile_options(binary_type, execution):
    return []


def deduce_target(execution):
    cwd = execution[0]
    cmd = execution[1]

    binary = cmd[0]
    binary_type = detect_binary_type(binary)
    output, target_type = detect_output(binary_type, cmd)
    if output is None:
        return None
    sources = detect_sources(execution)

    include_dirs = detect_include_dirs(binary_type, execution)
    definitions = detect_definitions(binary_type, execution)
    link_libraries = detect_link_libraries(binary_type, execution)
    compile_options = detect_compile_options(binary_type, execution)
    objects = detect_objects(binary_type, execution)

    target = Target(binary_type, output, target_type, sources, cwd, cmd, include_dirs,
                    definitions, link_libraries, compile_options, objects)

    return target


def deduce_targets(executions):
    targets = []
    for e in executions:
        target = deduce_target(e)
        if target is not None:
            targets.append(target)
    return targets


def get_better_target(t1, t2):
    def is_compiler(target):
        return target.binary_type is BinaryType.CPP_COMPILER or \
               target.binary_type is BinaryType.C_COMPILER
    if is_compiler(t1) and is_compiler(t2):
        raise NotImplementedError
    elif is_compiler(t1):
        return t1
    elif is_compiler(t2):
        return t2
    else:
        raise NotImplementedError


def remove_duplicates(targets):
    target_names = (target.name for target in targets)
    new_targets = {n: None for n in target_names}
    for target in targets:
        prev_target = new_targets[target.name]
        if prev_target is None:
            new_targets[target.name] = target
        else:
            new_targets[target.name] = get_better_target(target, prev_target)

    return [new_targets[name] for name in new_targets]


def write_cmake_header(cmake):
    cmake.write(dedent("""
        cmake_minimum_required(VERSION 3.12)
    
        #project(converted_with_gpu_mpi LANGUAGES C CXX CUDA)
        project(converted_with_gpu_mpi LANGUAGES C CXX)
    
        # set(CMAKE_CUDA_FLAGS 
        #     "${CMAKE_CUDA_FLAGS} \
        #     -gencode arch=compute_75,code=sm_75")
    
        #include(${SCRIPTDIR}/../gpu_libs/gpu_libs-exports.cmake)
    
        #set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
        file(WRITE null.cpp "")
    """))


def write_cmake_target(cmake, target):
    cmake.write("\n# directory: {cwd}\n".format(cwd=target.cwd))
    cmake.write("# command: {cmd}\n".format(cmd=target.cmd))
    if target.target_type is TargetType.EXECUTABLE or \
            target.target_type is TargetType.STATIC_LIB or \
            target.target_type is TargetType.SHARED_LIB:

        shared_or_static = ""
        if target.target_type == TargetType.STATIC_LIB:
            shared_or_static = "STATIC"
        elif target.target_type == TargetType.SHARED_LIB:
            shared_or_static = "SHARED"

        cmake.write("{add_target}({target} {shared_or_static} {sources})\n".format(
            add_target="add_executable" if target.target_type == TargetType.EXECUTABLE else "add_library",
            target=target.name,
            shared_or_static=shared_or_static,
            sources=" ".join(target.sources) if len(target.sources) > 0 else "null.cpp"
        ))

        if len(target.include_dirs) > 0:
            cmake.write("target_include_directories({target} PUBLIC {include_dirs})\n".format(
                target=target.name, include_dirs=" ".join(target.include_dirs)
            ))

        if len(target.link_libraries) > 0:
            cmake.write("target_link_libraries({target} PUBLIC {link_libraries})\n".format(
                target=target.name, link_libraries=" ".join(target.link_libraries)
            ))
        if len(target.objects) > 0:
            cmake.write("target_link_libraries({target} PUBLIC {objects})\n".format(
                target=target.name, objects=" ".join(target.objects)
            ))

        if len(target.definitions) > 0:
            cmake.write("target_compile_definitions({target} PUBLIC {compile_definitions})\n".format(
                target=target.name, compile_definitions=" ".join(target.definitions)
            ))

        if len(target.compile_options) > 0:
            cmake.write("target_compile_options({target} PUBLIC {compile_options})\n".format(
                target=target.name, compile_options=" ".join(target.compile_options)
            ))


def generate_cmake():
    import json
    with open("execution_log.txt", "r") as f:
        executions = json.load(f)
    targets = deduce_targets(executions)

    targets = remove_duplicates(targets)

    Path("gpu_mpi_project").mkdir(parents=True, exist_ok=True)
    with open("gpu_mpi_project/CMakeLists.txt", "w") as cmake:
        write_cmake_header(cmake)
        for target in targets:
            write_cmake_target(cmake, target)


if __name__ == "__main__":
    generate_cmake()
