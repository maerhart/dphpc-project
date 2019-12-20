cmake_minimum_required(VERSION 3.12)

project(converter)

include(FetchContent)

FetchContent_Declare(llvm
    URL https://releases.llvm.org/8.0.0/llvm-8.0.0.src.tar.xz
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm
)

FetchContent_GetProperties(llvm)
if(NOT llvm_POPULATED)
    FetchContent_Populate(llvm)
endif()

FetchContent_Declare(clang
    URL https://releases.llvm.org/8.0.0/cfe-8.0.0.src.tar.xz
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm/tools/clang
)

FetchContent_GetProperties(clang)
if(NOT clang_POPULATED)
    FetchContent_Populate(clang)
endif()

function(add_llvm)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS_SAVED}")

    # it is essential to not use CACHE for CMAKE_BUILD_TYPE, but use it for BUILD_SHARED_LIBS
    set(CMAKE_BUILD_TYPE Release)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE) # enables shared shibraries to link with

    set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "" FORCE) # not needed since we use only frontend
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE) # otherwise it is impossible to inherit from llvm classess

    add_subdirectory(${llvm_SOURCE_DIR} ${llvm_BINARY_DIR} EXCLUDE_FROM_ALL)

    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}" CACHE BOOL "" FORCE)
endfunction()

add_llvm()

add_executable(converter converter.cpp)

# because LLVM itself doesn't define include dirs
target_include_directories(converter PRIVATE
    ${llvm_SOURCE_DIR}/include
    ${clang_SOURCE_DIR}/include
    ${llvm_BINARY_DIR}/tools/clang/include
    ${llvm_BINARY_DIR}/include
)

# it makes headers visible for libTooling based tools in build dir
add_dependencies(converter clang-headers)

target_link_libraries(converter PRIVATE
    LLVMOption
    LLVMSupport
    clangAST
    clangASTMatchers
    clangBasic
    clangDriver
    clangFrontend
    clangLex
    clangRewrite
    clangRewriteFrontend
    clangSerialization
    clangStaticAnalyzerFrontend
    clangTooling
    clangSema
)

set_target_properties(converter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${llvm_BINARY_DIR}/bin"
)
