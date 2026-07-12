#pragma once

#include "uiparsercv/pipeline/ui_element.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace uiparsercv::pipeline {

struct VisualContainerOptions {
  bool enabled{true};
  float min_area_ratio{0.004F};
  float max_area_ratio{0.92F};
  float min_rectangularity{0.68F};
  float duplicate_iou{0.88F};
  int min_side{28};
};

[[nodiscard]] std::vector<UiElementCandidate> detect_visual_containers(
    const cv::Mat& bgr_image,
    const std::vector<UiElementCandidate>& existing,
    VisualContainerOptions options = {});

} // namespace uiparsercv::pipeline
