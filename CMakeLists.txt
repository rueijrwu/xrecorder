cmake_minimum_required(VERSION 3.18)

# Allow unsupported GCC versions for nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(ximea_highspeed_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 gstreamer-cuda-1.0)

# Find CLI11
find_package(CLI11 REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# XIMEA Paths
set(XIMEA_INCLUDE_DIR "/usr/include") 
set(XIMEA_LIB_DIR "/opt/XIMEA/lib")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GST_INCLUDE_DIRS}
    ${XIMEA_INCLUDE_DIR}
)

# 1. Library target
add_library(ximea_core STATIC
    src/ximea_capture.cpp
    src/gst_recorder.cpp
    src/gst_previewer.cpp
    src/ximea_manager.cpp
    src/cuda_utils.cu
    src/encoder_lane.cpp
    src/gop_scheduler.cpp
    src/ordered_bitstream_serializer.cpp
    src/multi_nv_recorder.cpp
)
set_target_properties(ximea_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ximea_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ximea_core
    PUBLIC
    ${GST_LIBRARIES}
    -L${XIMEA_LIB_DIR} -lm3api
    CUDA::cudart
    CUDA::cuda_driver
    pthread
    dl
)

# 2. CLI App target
add_executable(ximea_cli src/main.cpp)
target_link_libraries(ximea_cli
    PRIVATE
    ximea_core
    CLI11::CLI11
    CUDA::cuda_driver
)
# Force the linker to keep CUDA driver symbols and export them for GStreamer's dynamic loader
target_link_options(ximea_cli PRIVATE "LINKER:--no-as-needed" "LINKER:--export-dynamic")

# 3. Python Module target
pybind11_add_module(ximea_py python/binding/bindings.cpp)
target_link_libraries(ximea_py 
    PRIVATE 
    ximea_core
    CUDA::cuda_driver
)
target_link_options(ximea_py PRIVATE "LINKER:--no-as-needed")

# 4. Example Usage target
add_executable(ximea_cpp_example src/example_usage.cpp)
target_link_libraries(ximea_cpp_example
    PRIVATE
    ximea_core
)
