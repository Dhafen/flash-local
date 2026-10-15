cmake_minimum_required(VERSION 3.20)
project(FlashLocalReference LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
if(NOT LLAMA_SOURCE)
  message(FATAL_ERROR "LLAMA_SOURCE must point to the pinned reference checkout")
endif()
add_subdirectory("${LLAMA_SOURCE}" llama)
add_executable(flash-local-reference main.cpp)
target_link_libraries(flash-local-reference PRIVATE llama)
add_executable(flash-local-tokenizer-reference tokenizer.cpp)
add_library(flash-local-tokenizer-runtime STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../Sources/CLlamaTokenizer/FlashLlamaTokenizer.cpp"
  "${LLAMA_SOURCE}/common/json.cpp"
  "${LLAMA_SOURCE}/common/unicode.cpp"
  "${LLAMA_SOURCE}/common/jinja/lexer.cpp"
  "${LLAMA_SOURCE}/common/jinja/parser.cpp"
  "${LLAMA_SOURCE}/common/jinja/runtime.cpp"
  "${LLAMA_SOURCE}/common/jinja/string.cpp"
  "${LLAMA_SOURCE}/common/jinja/value.cpp")
target_include_directories(flash-local-tokenizer-runtime PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../Sources/CLlamaTokenizer/include")
target_include_directories(flash-local-tokenizer-runtime PRIVATE
  "${LLAMA_SOURCE}/common" "${LLAMA_SOURCE}/src" "${LLAMA_SOURCE}/vendor")
target_link_libraries(flash-local-tokenizer-runtime PUBLIC llama)
target_link_libraries(flash-local-tokenizer-reference PRIVATE flash-local-tokenizer-runtime)
target_include_directories(flash-local-tokenizer-reference PRIVATE
  "${LLAMA_SOURCE}/common" "${LLAMA_SOURCE}/src" "${LLAMA_SOURCE}/vendor")
