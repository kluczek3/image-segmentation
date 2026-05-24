cmake_minimum_required(VERSION 3.18)
project(ImageSegmentationCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "all")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

find_package(Qt6 COMPONENTS Core Gui Widgets Concurrent REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

include_directories(include)

add_executable(${PROJECT_NAME} WIN32
    src/main.cpp
    src/StartScreen.cpp
    src/ResultScreen.cpp
    src/CoreManager.cpp
    src/CPU_Algorithms.cpp
    src/GPU_Algorithms.cu
    include/StartScreen.h
    include/ResultScreen.h
    include/SegmentationAlgorithm.h
    include/ImageConverter.h
    include/CoreManager.h
    include/CPU_Algorithms.h
    include/GPU_Algorithms.h
    ui/StartScreen.ui
    ui/ResultScreen.ui
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    Qt6::Widgets 
    Qt6::Concurrent 
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS "${_qt_bin_dir}")

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env PATH="${_qt_bin_dir}" ${WINDEPLOYQT_EXECUTABLE} "$<TARGET_FILE:${PROJECT_NAME}>"
    )
endif()