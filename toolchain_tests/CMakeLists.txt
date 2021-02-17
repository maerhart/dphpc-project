cmake_minimum_required(VERSION 3.12)
project(Foo)

find_program (BASH_PROGRAM bash)


# Test Pi computation example from mpich

configure_file(pi/cpi.c pi/cpi.c COPYONLY)
configure_file(pi/Makefile pi/Makefile COPYONLY)
configure_file(pi/test.sh pi/test.sh COPYONLY)

add_test(
    NAME pi_test
    COMMAND ${BASH_PROGRAM} test.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pi"
)

# Test global var support 

configure_file(global_var/global_var.c global_var/global_var.c COPYONLY)
configure_file(global_var/Makefile global_var/Makefile COPYONLY)
configure_file(global_var/test.sh global_var/test.sh COPYONLY)

add_test(
    NAME global_var_test
    COMMAND ${BASH_PROGRAM} test.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/global_var"
)

# Test NAS Parallel Benchmarks

file(COPY npb/NPB3.4.1 DESTINATION npb)
configure_file(npb/dt_test.sh npb/dt_test.sh COPYONLY)
configure_file(npb/is_test.sh npb/is_test.sh COPYONLY)

add_test(
    NAME npb_is_test
    COMMAND ${BASH_PROGRAM} is_test.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/npb"
)

add_test(
    NAME npb_dt_test
    COMMAND ${BASH_PROGRAM} dt_test.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/npb"
)
