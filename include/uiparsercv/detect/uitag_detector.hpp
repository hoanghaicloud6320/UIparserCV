#pragma once

#include "uiparsercv/infer/onnx_session.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <filesystem>
#include <string_view>
#include <vector>

namespace uiparsercv::detect {

struct UitagDetection {
  cv::Rect2f box;
  float score{};
  int class_id{};
};

struct UitagOptions {
  int tile_size{640};
  int tile_step{512};
  float confidence_threshold{0.25F};
  float tile_nms_iou{0.70F};
  float cross_tile_nms_iou{0.50F};
};

struct UitagLetterbox {
  float scale{1.0F};
  int left{};
  int top{};
};

struct UitagResult {
  int tile_count{};
  // Per-tile detections before cross-tile deduplication, retained for diagnostics.
  std::vector<UitagDetection> raw_detections;
  std::vector<UitagDetection> detections;
};

class UitagDetector {
public:
  explicit UitagDetector(
      const std::filesystem::path& model_path,
      UitagOptions options = {});

  [[nodiscard]] UitagResult detect(const cv::Mat& bgr_image);
  [[nodiscard]] const UitagOptions& options() const noexcept;

  [[nodiscard]] static std::vector<int> tile_origins(
      int extent, int tile_size, int tile_step);
  [[nodiscard]] static std::vector<UitagDetection> decode_output(
      const infer::Tensor& output,
      const UitagLetterbox& letterbox,
      const cv::Size& tile_size,
      const cv::Point& tile_origin,
      const cv::Size& image_size,
      float confidence_threshold);
  [[nodiscard]] static std::vector<UitagDetection> nms(
      std::vector<UitagDetection> detections,
      float iou_threshold,
      bool class_aware);
  [[nodiscard]] static std::string_view class_name(int class_id);

private:
  infer::OnnxSession session_;
  UitagOptions options_;
};

}  // namespace uiparsercv::detect
