cmake_minimum_required(VERSION 3.12)

project(converter)

find_package(LLVM REQUIRED)

find_package(Clang REQUIRED)

add_executable(converter converter.cpp)

target_include_directories(converter PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(converter PUBLIC ${CLANG_INCLUDE_DIRS})


target_link_libraries(converter PRIVATE
    LLVMOption
    LLVMSupport
    LLVMFrontendOpenMP # I have to add this to fix link issue in clang 11.0.1
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

set(LLVM_BUILTIN_HEADERS "${LLVM_INSTALL_PREFIX}/lib/clang/${LLVM_PACKAGE_VERSION}/include")
if(NOT IS_DIRECTORY ${LLVM_BUILTIN_HEADERS} OR NOT EXISTS ${LLVM_BUILTIN_HEADERS})
    message(FATAL_ERROR "LLVM_BUILTIN_HEADERS are not found at ${LLVM_BUILTIN_HEADERS}")
endif()

target_compile_definitions(converter PUBLIC "LLVM_BUILTIN_HEADERS=${LLVM_BUILTIN_HEADERS}")
