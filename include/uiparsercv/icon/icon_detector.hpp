#pragma once

#include "uiparsercv/common/types.hpp"
#include "uiparsercv/infer/onnx_session.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <vector>

namespace uiparsercv::icon {

struct IconDetectorOptions {
  std::filesystem::path model_path{"models/icon_detect/model.onnx"};
  int input_size{640};
  float confidence_threshold{0.05F};
  float nms_threshold{0.1F};
};

class IconDetector {
public:
  explicit IconDetector(IconDetectorOptions options = {});

  [[nodiscard]] std::vector<Detection> detect(const cv::Mat& bgr_image);

private:
  IconDetectorOptions options_;
  infer::OnnxSession session_;
};

} // namespace uiparsercv::icon
