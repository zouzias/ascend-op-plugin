cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(OPPLUGINUT CXX C)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

FILE(GLOB OP_PLUGIN_TEST_SRCS CONFIGURE_DEPENDS
    *.cpp)

# Set installed PyTorch dir
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/distributed)
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

# Set installed torch_npu dir
if(DEFINED TORCH_NPU_INSTALL_DIR)
  include_directories(${TORCH_NPU_INSTALL_DIR}/include)
else()
  message(FATAL_ERROR, "Cannot find installed torch_npu directory")
endif()

if(DEFINED BUILD_GTEST)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pytorch_ut/third_party/googletest gtest_dir)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../pytorch_ut/third_party/googletest/googletest/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

add_executable(ops_cpp_utest ${OP_PLUGIN_TEST_SRCS})

if(DEFINED BUILD_GTEST)
  target_link_libraries(ops_cpp_utest PUBLIC gtest_main gtest)
endif()

# link pytorch so
if(DEFINED PYTORCH_INSTALL_DIR)
  target_link_libraries(ops_cpp_utest PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch.so)
  target_link_libraries(ops_cpp_utest PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)
  target_link_libraries(ops_cpp_utest PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
else()
  message(FATAL_ERROR, "Cannot link libtorch.so, libc10.so or libtorch_cpu.so")
endif()

# link torch_npu so
if(DEFINED TORCH_NPU_INSTALL_DIR)
  target_link_libraries(ops_cpp_utest PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../pytorch_ut/build/Release/packages/torch_npu/lib/libtorch_npu.so)
else()
  message(FATAL_ERROR, "Cannot link libtorch_npu.so")
endif()
