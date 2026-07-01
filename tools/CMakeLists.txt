add_executable(ui_tree_probe ui_tree_probe.cpp)
target_link_libraries(ui_tree_probe PRIVATE uiparsercv_core)
add_test(NAME ui_tree_probe COMMAND ui_tree_probe)

add_executable(model_file_probe model_file_probe.cpp)
target_compile_definitions(model_file_probe
  PRIVATE
    UIPARSERCV_ICON_MODEL="${UIPARSERCV_ICON_MODEL}"
    UIPARSERCV_OCR_DET_MODEL="${UIPARSERCV_OCR_DET_MODEL}"
    UIPARSERCV_OCR_REC_MODEL="${UIPARSERCV_OCR_REC_MODEL}"
)

if(TARGET PkgConfig::OpenCV)
  add_executable(opencv_probe opencv_probe.cpp)
  target_link_libraries(opencv_probe PRIVATE PkgConfig::OpenCV)
  target_compile_features(opencv_probe PRIVATE cxx_std_20)

  add_executable(synthetic_ui_corpus synthetic_ui_corpus.cpp)
  target_link_libraries(synthetic_ui_corpus PRIVATE PkgConfig::OpenCV)
  target_compile_features(synthetic_ui_corpus PRIVATE cxx_std_20)
  add_test(
    NAME synthetic_ui_corpus
    COMMAND synthetic_ui_corpus
      "${CMAKE_BINARY_DIR}/testdata/synthetic_ui"
      36
  )
else()
  message(STATUS "OpenCV not found via pkg-config; opencv_probe will not be built")
endif()

if(TARGET PkgConfig::ONNXRuntime)
  add_executable(onnxruntime_probe onnxruntime_probe.cpp)
  target_link_libraries(onnxruntime_probe PRIVATE PkgConfig::ONNXRuntime)
  target_compile_features(onnxruntime_probe PRIVATE cxx_std_20)
  add_test(NAME onnxruntime_probe COMMAND onnxruntime_probe)
  uiparsercv_copy_onnxruntime(onnxruntime_probe)
else()
  message(STATUS "ONNX Runtime not found via pkg-config; onnxruntime_probe will not be built")
endif()

if(TARGET uiparsercv_infer)
  add_executable(component_probe component_probe.cpp)
  target_link_libraries(component_probe PRIVATE uiparsercv_infer)
  target_compile_features(component_probe PRIVATE cxx_std_20)
  target_compile_definitions(component_probe
    PRIVATE
      UIPARSERCV_ICON_MODEL="${UIPARSERCV_ICON_MODEL}"
      UIPARSERCV_OCR_DET_MODEL="${UIPARSERCV_OCR_DET_MODEL}"
      UIPARSERCV_OCR_REC_MODEL="${UIPARSERCV_OCR_REC_MODEL}"
      UIPARSERCV_OCR_REC_CONFIG="${UIPARSERCV_OCR_REC_CONFIG}"
  )
  uiparsercv_copy_onnxruntime(component_probe)
  add_test(NAME component_probe COMMAND component_probe)

  add_executable(model_combo_compare model_combo_compare.cpp)
  target_link_libraries(model_combo_compare PRIVATE uiparsercv_pipeline)
  target_compile_features(model_combo_compare PRIVATE cxx_std_20)
  target_compile_definitions(model_combo_compare
    PRIVATE
      UIPARSERCV_ICON_MODEL="${UIPARSERCV_ICON_MODEL}"
      UIPARSERCV_OCR_TINY_DET_MODEL="${UIPARSERCV_OCR_TINY_DET_MODEL}"
      UIPARSERCV_OCR_TINY_REC_MODEL="${UIPARSERCV_OCR_TINY_REC_MODEL}"
      UIPARSERCV_OCR_TINY_REC_CONFIG="${UIPARSERCV_OCR_TINY_REC_CONFIG}"
      UIPARSERCV_OCR_SMALL_DET_MODEL="${UIPARSERCV_OCR_SMALL_DET_MODEL}"
      UIPARSERCV_OCR_SMALL_REC_MODEL="${UIPARSERCV_OCR_SMALL_REC_MODEL}"
      UIPARSERCV_OCR_SMALL_REC_CONFIG="${UIPARSERCV_OCR_SMALL_REC_CONFIG}"
  )
  uiparsercv_copy_onnxruntime(model_combo_compare)
endif()
