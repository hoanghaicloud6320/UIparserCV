#pragma once

#include "uiparsercv/pipeline/ui_element.hpp"

#include <vector>

namespace uiparsercv::pipeline {

struct BoundarylessGroupOptions {
  bool enabled{true};
  int min_repetitions{2};
  float max_normalized_gap{0.80F};
  float min_axis_overlap{0.55F};
  float alignment_tolerance{0.25F};
  float isolation_ratio{1.65F};
};

[[nodiscard]] std::vector<UiElementCandidate> detect_boundaryless_groups(
    const std::vector<UiElementCandidate>& candidates,
    int image_width,
    int image_height,
    BoundarylessGroupOptions options = {});

} // namespace uiparsercv::pipeline
