cmake_minimum_required(VERSION 3.24)

project(UIparserCV
  VERSION 0.1.0
  DESCRIPTION "Parse UI screenshots into structured UI trees"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(UIPARSERCV_BUILD_PROBES "Build independent component probe executables" ON)

set(UIPARSERCV_ICON_MODEL "${CMAKE_SOURCE_DIR}/models/icon_detect/model.onnx"
  CACHE FILEPATH "Path to OmniParser icon detection ONNX model")
set(UIPARSERCV_OCR_DET_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_tiny_det.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 detection model")
set(UIPARSERCV_OCR_REC_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_small_rec.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 recognition model")
set(UIPARSERCV_OCR_REC_CONFIG "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_small_rec.yml"
  CACHE FILEPATH "Path to PP-OCRv6 recognition YAML config")
set(UIPARSERCV_OCR_TINY_DET_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_tiny_det.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 tiny detection model")
set(UIPARSERCV_OCR_TINY_REC_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_tiny_rec.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 tiny recognition model")
set(UIPARSERCV_OCR_TINY_REC_CONFIG "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_tiny_rec.yml"
  CACHE FILEPATH "Path to PP-OCRv6 tiny recognition YAML config")
set(UIPARSERCV_OCR_SMALL_DET_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_small_det.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 small detection model")
set(UIPARSERCV_OCR_SMALL_REC_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_small_rec.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 small recognition model")
set(UIPARSERCV_OCR_SMALL_REC_CONFIG "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_small_rec.yml"
  CACHE FILEPATH "Path to PP-OCRv6 small recognition YAML config")
set(UIPARSERCV_OCR_MEDIUM_DET_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_medium_det.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 medium detection model")
set(UIPARSERCV_OCR_MEDIUM_REC_MODEL "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_medium_rec.onnx"
  CACHE FILEPATH "Path to PP-OCRv6 medium recognition model")
set(UIPARSERCV_OCR_MEDIUM_REC_CONFIG "${CMAKE_SOURCE_DIR}/models/ocr/ppocrv6_medium_rec.yml"
  CACHE FILEPATH "Path to PP-OCRv6 medium recognition YAML config")

include(GNUInstallDirs)
include(CTest)

add_library(uiparsercv_core
  src/tree/box_tree.cpp
)

target_include_directories(uiparsercv_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(uiparsercv_core PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(uiparsercv_core PRIVATE /W4)
else()
  target_compile_options(uiparsercv_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(OpenCV QUIET IMPORTED_TARGET opencv4)
  pkg_check_modules(ONNXRuntime QUIET IMPORTED_TARGET libonnxruntime)
endif()

function(uiparsercv_copy_onnxruntime target_name)
  if(NOT TARGET PkgConfig::ONNXRuntime)
    return()
  endif()

  find_program(UIPARSERCV_ONNXRUNTIME_DLL onnxruntime.dll
    HINTS "${ONNXRuntime_LIBDIR}/../bin" "${ONNXRuntime_PREFIX}/bin")
  find_program(UIPARSERCV_ONNXRUNTIME_PROVIDERS_DLL onnxruntime_providers_shared.dll
    HINTS "${ONNXRuntime_LIBDIR}/../bin" "${ONNXRuntime_PREFIX}/bin")

  if(UIPARSERCV_ONNXRUNTIME_DLL)
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${UIPARSERCV_ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:${target_name}>")
  endif()

  if(UIPARSERCV_ONNXRUNTIME_PROVIDERS_DLL)
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${UIPARSERCV_ONNXRUNTIME_PROVIDERS_DLL}"
        "$<TARGET_FILE_DIR:${target_name}>")
  endif()
endfunction()

if(TARGET PkgConfig::OpenCV AND TARGET PkgConfig::ONNXRuntime)
  add_library(uiparsercv_infer
    src/icon/icon_detector.cpp
    src/infer/onnx_session.cpp
    src/ocr/ocr_detector.cpp
    src/ocr/ocr_recognizer.cpp
    src/ocr/text_normalizer.cpp
    src/pipeline/ui_element.cpp
    third_party/clipper/clipper.cpp
  )

  target_include_directories(uiparsercv_infer
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/clipper
  )

  target_compile_features(uiparsercv_infer PUBLIC cxx_std_20)
  target_link_libraries(uiparsercv_infer
    PUBLIC
      PkgConfig::OpenCV
      PkgConfig::ONNXRuntime
  )

  if(MSVC)
    target_compile_options(uiparsercv_infer PRIVATE /W4)
  else()
    target_compile_options(uiparsercv_infer PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(third_party/clipper/clipper.cpp
      PROPERTIES COMPILE_OPTIONS "-w")
  endif()

  add_library(uiparsercv_pipeline
    src/export/debug_overlay.cpp
    src/export/json_exporter.cpp
    src/pipeline/tree_grouper.cpp
    src/pipeline/pipeline_runner.cpp
  )

  target_include_directories(uiparsercv_pipeline
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  target_compile_features(uiparsercv_pipeline PUBLIC cxx_std_20)
  target_link_libraries(uiparsercv_pipeline
    PUBLIC
      uiparsercv_core
      uiparsercv_infer
  )

  if(MSVC)
    target_compile_options(uiparsercv_pipeline PRIVATE /W4)
  else()
    target_compile_options(uiparsercv_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_executable(uiparsercv_pipeline_cli
    src/app/pipeline_cli.cpp
  )
  set_target_properties(uiparsercv_pipeline_cli
    PROPERTIES OUTPUT_NAME uiparsercv_pipeline)
  target_link_libraries(uiparsercv_pipeline_cli PRIVATE uiparsercv_pipeline)
  target_compile_definitions(uiparsercv_pipeline_cli
    PRIVATE
      UIPARSERCV_ICON_MODEL="${UIPARSERCV_ICON_MODEL}"
      UIPARSERCV_OCR_DET_MODEL="${UIPARSERCV_OCR_DET_MODEL}"
      UIPARSERCV_OCR_REC_MODEL="${UIPARSERCV_OCR_REC_MODEL}"
      UIPARSERCV_OCR_REC_CONFIG="${UIPARSERCV_OCR_REC_CONFIG}"
  )
  uiparsercv_copy_onnxruntime(uiparsercv_pipeline_cli)
else()
  message(STATUS "OpenCV and ONNX Runtime are required for uiparsercv_infer")
endif()

if(UIPARSERCV_BUILD_PROBES)
  add_subdirectory(tools)
endif()
