cmake_minimum_required(VERSION 3.12)
project(Foo)

find_program (BASH_PROGRAM bash)


configure_file(pi/cpi.c pi/cpi.c COPYONLY)
configure_file(pi/Makefile pi/Makefile COPYONLY)
configure_file(pi/test.sh pi/test.sh COPYONLY)

add_test(
    NAME pi_test
    COMMAND ${BASH_PROGRAM} test.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pi"
)


configure_file(global_var/global_var.c global_var/global_var.c COPYONLY)
configure_file(global_var/Makefile global_var/Makefile COPYONLY)
configure_file(global_var/test.sh global_var/test.sh COPYONLY)

add_test(
    NAME global_var_test
    COMMAND ${BASH_PROGRAM} test.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/global_var"
)
