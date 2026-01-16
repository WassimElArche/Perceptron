cmake_minimum_required(VERSION 3.16)
project(peceptron C)

set(CMAKE_C_STANDARD 11)

# Trouver raylib
find_package(raylib REQUIRED)

add_executable(peceptron
    main.c
    dataset.c
    perceptron.c
    visual.c
)

target_include_directories(peceptron PRIVATE .)
target_link_libraries(peceptron raylib)