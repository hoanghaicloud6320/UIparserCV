#pragma once

#include "uiparsercv/pipeline/ui_element.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace uiparsercv::pipeline {

struct LineRectOptions {
  bool enabled{true};
  int coordinate_tolerance{4};
  int merge_gap{14};
  int min_side{28};
  float min_area_ratio{0.003F};
  float max_area_ratio{0.90F};
  float min_side_support{0.66F};
  float duplicate_iou{0.88F};
  float max_interior_edge_density{0.20F};
  int max_axis_lines{72};
  int max_proposals{128};
};

[[nodiscard]] std::vector<UiElementCandidate> detect_line_rects(
    const cv::Mat& bgr_image,
    const std::vector<UiElementCandidate>& existing,
    LineRectOptions options = {});

} // namespace uiparsercv::pipeline
