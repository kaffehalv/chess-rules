cmake_minimum_required(VERSION 3.18)
project(chess-rules LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

option(CR_BUILD_TESTS "Build the Catch2 tests" ON)
option(CR_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer"
       OFF)
set(CR_CHESS_LIBRARY_DIR
    ""
    CACHE
      PATH
      "Directory that contains chess.hpp. Default: the git submodule in external/chess-library/include"
)

include(FetchContent)

# ---------------------------------------------------------------------------------------------
# chess-library (header only), a git submodule in external/chess-library
# ---------------------------------------------------------------------------------------------
if(CR_CHESS_LIBRARY_DIR)
  set(CR_CHESS_INCLUDE_DIR "${CR_CHESS_LIBRARY_DIR}")
else()
  set(CR_CHESS_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/external/chess-library/include")

  # A clone made without --recurse-submodules: fetch the submodule now.
  if(NOT EXISTS "${CR_CHESS_INCLUDE_DIR}/chess.hpp")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules")
      message(STATUS "chess-library is missing, initialising the git submodule")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
                external/chess-library
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE CR_SUBMODULE_RESULT)
      if(NOT CR_SUBMODULE_RESULT EQUAL 0)
        message(
          WARNING
            "git submodule update failed (exit code ${CR_SUBMODULE_RESULT})")
      endif()
    endif()
  endif()
endif()

if(NOT EXISTS "${CR_CHESS_INCLUDE_DIR}/chess.hpp")
  message(
    FATAL_ERROR
      "chess.hpp not found in ${CR_CHESS_INCLUDE_DIR}\n"
      "Run:  git submodule update --init external/chess-library\n"
      "or pass -DCR_CHESS_LIBRARY_DIR=<directory containing chess.hpp>")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------------------------
# Header-only library: include/chess-rules/*.hpp
# ---------------------------------------------------------------------------------------------
add_library(CR_lib INTERFACE)
target_include_directories(
  CR_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${CR_CHESS_INCLUDE_DIR}>)
target_compile_features(CR_lib INTERFACE cxx_std_17)
target_link_libraries(CR_lib INTERFACE Threads::Threads)

set(CR_WARNINGS $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra
                -Wpedantic>)
set(CR_SANITIZER_FLAGS
    $<$<AND:$<BOOL:${CR_SANITIZE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fsanitize=address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer>)

# ---------------------------------------------------------------------------------------------
# Command line tool
# ---------------------------------------------------------------------------------------------
add_executable(chess-rules src/main.cpp)
target_link_libraries(chess-rules PRIVATE CR_lib)
target_compile_options(chess-rules PRIVATE ${CR_WARNINGS} ${CR_SANITIZER_FLAGS})
target_link_options(chess-rules PRIVATE ${CR_SANITIZER_FLAGS})

install(TARGETS chess-rules RUNTIME DESTINATION bin)
install(DIRECTORY include/chess-rules DESTINATION include)

# ---------------------------------------------------------------------------------------------
# Tests (Catch2 v3)
# ---------------------------------------------------------------------------------------------
if(CR_BUILD_TESTS)
  enable_testing()

  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.4)
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  endif()

  add_executable(CR_tests tests/test_chess960.cpp tests/test_rules.cpp
                          tests/test_builders.cpp tests/test_generator.cpp)
  target_link_libraries(CR_tests PRIVATE CR_lib Catch2::Catch2WithMain)
  target_compile_options(CR_tests PRIVATE ${CR_WARNINGS} ${CR_SANITIZER_FLAGS})
  target_link_options(CR_tests PRIVATE ${CR_SANITIZER_FLAGS})

  include(CTest)
  include(Catch)
  catch_discover_tests(CR_tests)
endif()
