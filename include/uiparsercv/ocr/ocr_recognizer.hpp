#pragma once

#include "uiparsercv/common/types.hpp"
#include "uiparsercv/infer/onnx_session.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace uiparsercv::ocr {

struct OcrRecognizerOptions {
  std::filesystem::path model_path{"models/ocr/rec.onnx"};
  std::filesystem::path config_path{"models/ocr/rec.yml"};
  int image_channels{3};
  int image_height{48};
  int image_width{320};
  int max_width{3200};
};

class OcrTextRecognizer {
public:
  explicit OcrTextRecognizer(OcrRecognizerOptions options = {});

  [[nodiscard]] TextRecognition recognize(const cv::Mat& bgr_crop);
  [[nodiscard]] const std::vector<std::string>& character_list() const;

private:
  OcrRecognizerOptions options_;
  infer::OnnxSession session_;
  std::vector<std::string> characters_;
};

} // namespace uiparsercv::ocr

