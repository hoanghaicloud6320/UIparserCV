#pragma once

#include "uiparsercv/common/types.hpp"
#include "uiparsercv/infer/onnx_session.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <vector>

namespace uiparsercv::ocr {

struct OcrDetectorOptions {
  std::filesystem::path model_path{"models/ocr/det.onnx"};
  int limit_side_len{736};
  int max_side_limit{4000};
  float threshold{0.2F};
  float box_threshold{0.4F};
  float unclip_ratio{1.4F};
  int max_candidates{3000};
};

class OcrTextDetector {
public:
  explicit OcrTextDetector(OcrDetectorOptions options = {});

  [[nodiscard]] std::vector<TextRegion> detect(const cv::Mat& bgr_image);
  [[nodiscard]] cv::Mat crop_region(const cv::Mat& bgr_image, const TextRegion& region) const;

private:
  OcrDetectorOptions options_;
  infer::OnnxSession session_;
};

} // namespace uiparsercv::ocr

