CMAKE_MINIMUM_REQUIRED(VERSION 3.7)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

SET(_SOURCE_DIR "src/")

PROJECT("vectorAdd")

################################################################################
# Find cupla
################################################################################

SET(cupla_ROOT "$ENV{CUPLA_ROOT}" CACHE STRING  "The location of the cupla library")

LIST(APPEND CMAKE_MODULE_PATH "${cupla_ROOT}")
FIND_PACKAGE("cupla" REQUIRED)

################################################################################
# Add library.
################################################################################

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "" "cpp" _FILES_SOURCE_CXX)

set(_INCLUDE_DIRECTORIES_PRIVATE ${_INCLUDE_DIR})

include_directories(${_INCLUDE_DIRECTORIES_PRIVATE})

# Always add all files to the target executable build call to add them to the build project.
cupla_add_executable(
    "vectorAdd"
    ${_FILES_SOURCE_CXX})

# Set the link libraries for this library (adds libs, include directories, defines and compile options).
target_link_libraries(
    "vectorAdd"
    PUBLIC ${cupla_LIBRARIES})
