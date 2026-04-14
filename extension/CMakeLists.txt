cmake_minimum_required(VERSION 3.16)
project(hybrid_search CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SQLite3
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SQLITE3 sqlite3)
endif()

if(NOT SQLITE3_FOUND)
    find_path(SQLITE3_INCLUDE_DIRS sqlite3.h)
    find_library(SQLITE3_LIBRARIES NAMES sqlite3)
endif()

# shared library
add_library(hybrid_search SHARED hybrid_search.cpp)

target_include_directories(hybrid_search PRIVATE
    ${SQLITE3_INCLUDE_DIRS}
)

# Do NOT link libsqlite3: the extension receives the sqlite3_api_routines
# pointer at load time via SQLITE_EXTENSION_INIT2.
# target_link_libraries(hybrid_search PRIVATE ${SQLITE3_LIBRARIES})

# remove the default "lib" prefix so output is hybrid_search.so
set_target_properties(hybrid_search PROPERTIES PREFIX "")

install(TARGETS hybrid_search DESTINATION lib)
