#pragma once

#include "uiparsercv/common/types.hpp"

#include <string>
#include <vector>

namespace uiparsercv::pipeline {

enum class UiElementKind {
  Icon,
  Text
};

struct UiElementCandidate {
  UiElementKind kind{};
  RectF box{};
  float detection_score{};
  std::string text;
  std::string raw_text;
  std::string normalized_text;
  float text_confidence{};
  bool interactive{};
  std::string source;
};

struct CandidateMergeOptions {
  float overlap_threshold{0.1F};
};

[[nodiscard]] std::vector<UiElementCandidate> build_candidates(
    const std::vector<Detection>& icons,
    const std::vector<TextRegion>& text_regions,
    const std::vector<TextRecognition>& recognized_text,
    CandidateMergeOptions options = {});

} // namespace uiparsercv::pipeline
