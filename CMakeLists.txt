cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

set(CUDA_ARCH "auto" CACHE STRING "CUDA architecture: auto, native, all, all-major, or a numeric list like 86 or 75;86")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS auto native all all-major 86 75 "75;86")

if(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(_cuda_arch_request "$ENV{CUDAARCHS}")
else()
    set(_cuda_arch_request "${CUDA_ARCH}")
endif()

if(_cuda_arch_request STREQUAL "auto")
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
elseif(_cuda_arch_request STREQUAL "all" OR _cuda_arch_request STREQUAL "all-major")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
        set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch_request}" CACHE STRING "CUDA architectures" FORCE)
    else()
        message(WARNING "CUDA_ARCH=${_cuda_arch_request} requires CMake >= 3.23. Falling back to compiler-default architecture selection.")
        unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    endif()
elseif(_cuda_arch_request STREQUAL "native")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
    else()
        message(WARNING "CUDA_ARCH=native requires CMake >= 3.24. Falling back to compiler-default architecture selection.")
        unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    endif()
else()
    set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch_request}" CACHE STRING "CUDA architectures" FORCE)
endif()

project(ImageSegmentationCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

message(STATUS "Requested CUDA architecture setting: ${_cuda_arch_request}")
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Final CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "Using compiler-default CUDA architecture selection.")
endif()

find_package(Qt6 COMPONENTS Core Gui Widgets Concurrent REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

include_directories(include)

add_executable(${PROJECT_NAME} WIN32
    src/main.cpp
    src/StartScreen.cpp
    src/CoreManager.cpp
    src/CPU_Algorithms.cpp
    src/GPU_Algorithms.cu

    include/StartScreen.h
    include/SegmentationAlgorithm.h
    include/ImageConverter.h
    include/CoreManager.h
    include/CPU_Algorithms.h
    include/GPU_Algorithms.h

    ui/StartScreen.ui
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Widgets
    Qt6::Concurrent
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(WIN32)
    get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS "${_qt_bin_dir}")

    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env PATH="${_qt_bin_dir}" ${WINDEPLOYQT_EXECUTABLE} "$<TARGET_FILE:${PROJECT_NAME}>"
        )
    endif()
endif()
