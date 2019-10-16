add_library(managed_memory
    managed_memory.cu
    hostdevicecommunicator.cu
    )
target_link_libraries(managed_memory PUBLIC gpumpi_common gpu_libc)

add_executable(test_managed_memory test_managed_memory.cu)
target_link_libraries(test_managed_memory PUBLIC managed_memory gpumpi_common)
