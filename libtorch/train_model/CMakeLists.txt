cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

file(GLOB example_app_src
     "*.h"
     "*.cpp"
)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(example-app ${example_app_src})
target_link_libraries(example-app "${OpenCV_LIBS}")
target_link_libraries(example-app "${TORCH_LIBRARIES}")

set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

