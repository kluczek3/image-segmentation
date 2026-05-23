cmake_minimum_required(VERSION 3.18)
project(ImageSegmentationCUDA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)

include_directories(include)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/StartScreen.cpp
    src/ResultScreen.cpp
    include/StartScreen.h
    include/ResultScreen.h
    ui/StartScreen.ui
    ui/ResultScreen.ui
)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets)