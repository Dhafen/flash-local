cmake_minimum_required(VERSION 3.24)
project(FlashOrnithLES LANGUAGES C CXX OBJC)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(flag COMMON TESTS EXAMPLES TOOLS SERVER APP UI MTMD)
  set(LLAMA_BUILD_${flag} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
if(NOT EXISTS "${OMNI_SOURCE}/flash-local-source.json")
  message(FATAL_ERROR "Use scripts/prepare_ornith_les.py for the pinned reviewed source")
endif()
add_subdirectory("${OMNI_SOURCE}" omni)
# Two llama runtimes must coexist. A dynamic framework exports ONLY flo_*;
# the old tokenizer/reference runtime retains its own statically linked symbols.
add_library(FlashOrnithLES SHARED FlashOrnithLES.cpp FlashOrnithLES.h MetalTelemetry.m)
target_include_directories(FlashOrnithLES PRIVATE
  "${OMNI_SOURCE}/src" "${OMNI_SOURCE}/vendor")
target_link_libraries(FlashOrnithLES PRIVATE llama "-framework Foundation" "-framework Metal")
target_link_options(FlashOrnithLES PRIVATE
  "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.txt")
set_target_properties(FlashOrnithLES PROPERTIES FRAMEWORK TRUE FRAMEWORK_VERSION A
  MACOSX_FRAMEWORK_IDENTIFIER com.dhafen.flashlocal.ornithles
  MACOSX_FRAMEWORK_BUNDLE_VERSION 1 MACOSX_FRAMEWORK_SHORT_VERSION_STRING 1.0.0
  PUBLIC_HEADER FlashOrnithLES.h BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_NAME_DIR "@rpath" XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  add_executable(ornith-les-control control.cpp)
  target_include_directories(ornith-les-control PRIVATE "${OMNI_SOURCE}/vendor")
  target_link_libraries(ornith-les-control PRIVATE FlashOrnithLES)
  add_executable(ornith-les-mixed-fixture "${CMAKE_CURRENT_SOURCE_DIR}/../../../Tests/ReferenceHarness/mixed.cpp" MetalTelemetry.m)
  target_link_libraries(ornith-les-mixed-fixture PRIVATE llama "-framework Metal")
endif()
