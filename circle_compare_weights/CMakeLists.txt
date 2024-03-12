cmake_minimum_required(VERSION 3.16)
project(compare_weights CXX)

set(ONE_INSTALL_DIR "" CACHE PATH "Path to ONE install dir")
if(NOT ONE_INSTALL_DIR)
  message(FATAL_ERROR "One install dir path is required")
endif()

set(CIRCLE_SCHEMA_DIR "" CACHE PATH "Circle schema dir")
if(NOT CIRCLE_SCHEMA_DIR)
  message(FATAL_ERROR "circle schema path is required")
endif()

add_executable(compare_weights main.cpp)

add_library(onert-core SHARED IMPORTED)
set_target_properties(onert-core PROPERTIES
  IMPORTED_LOCATION "${ONE_INSTALL_DIR}/lib/libonert_core.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ONE_INSTALL_DIR}/include/onert"
)

include_directories("${CIRCLE_SCHEMA_DIR}/include")

add_subdirectory("flatbuffers"
                 ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
                 EXCLUDE_FROM_ALL)

target_link_libraries(compare_weights PRIVATE flatbuffers)
target_link_libraries(compare_weights PRIVATE onert-core)
