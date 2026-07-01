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
else()
  message(STATUS "OpenCV not found via pkg-config; opencv_probe will not be built")
endif()

if(TARGET PkgConfig::ONNXRuntime)
  add_executable(onnxruntime_probe onnxruntime_probe.cpp)
  target_link_libraries(onnxruntime_probe PRIVATE PkgConfig::ONNXRuntime)
  target_compile_features(onnxruntime_probe PRIVATE cxx_std_20)
  add_test(NAME onnxruntime_probe COMMAND onnxruntime_probe)

  find_program(UIPARSERCV_ONNXRUNTIME_DLL onnxruntime.dll
    HINTS "${ONNXRuntime_LIBDIR}/../bin" "${ONNXRuntime_PREFIX}/bin")
  find_program(UIPARSERCV_ONNXRUNTIME_PROVIDERS_DLL onnxruntime_providers_shared.dll
    HINTS "${ONNXRuntime_LIBDIR}/../bin" "${ONNXRuntime_PREFIX}/bin")

  if(UIPARSERCV_ONNXRUNTIME_DLL)
    add_custom_command(TARGET onnxruntime_probe POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${UIPARSERCV_ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:onnxruntime_probe>")
  endif()

  if(UIPARSERCV_ONNXRUNTIME_PROVIDERS_DLL)
    add_custom_command(TARGET onnxruntime_probe POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${UIPARSERCV_ONNXRUNTIME_PROVIDERS_DLL}"
        "$<TARGET_FILE_DIR:onnxruntime_probe>")
  endif()
else()
  message(STATUS "ONNX Runtime not found via pkg-config; onnxruntime_probe will not be built")
endif()
