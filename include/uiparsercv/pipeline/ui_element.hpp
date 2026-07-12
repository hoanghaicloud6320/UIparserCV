#pragma once

#include "uiparsercv/common/types.hpp"
#include "uiparsercv/detect/uitag_detector.hpp"

#include <string>
#include <vector>

namespace uiparsercv::pipeline {

enum class UiElementKind {
  ModelProposal,
  Icon,
  Text,
  VisualContainer,
  InferredGroup
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
  std::string model_class;
};

struct CandidateAssociation {
  std::size_t proposal_index{};
  std::size_t text_index{};
  float confidence{};
  std::string relation;
};

struct CombinedCandidates {
  std::vector<UiElementCandidate> candidates;
  std::vector<CandidateAssociation> associations;
};

struct ModelOcrMergeOptions {
  float text_containment_threshold{0.80F};
  float near_identical_iou{0.80F};
  float legacy_icon_novelty_iou{0.30F};
};

struct CandidateMergeOptions {
  float overlap_threshold{0.1F};
  bool preserve_text_inside_icons{true};
};

[[nodiscard]] std::vector<UiElementCandidate> build_candidates(
    const std::vector<Detection>& icons,
    const std::vector<TextRegion>& text_regions,
    const std::vector<TextRecognition>& recognized_text,
    CandidateMergeOptions options = {});

[[nodiscard]] CombinedCandidates build_model_ocr_candidates(
    const std::vector<detect::UitagDetection>& model_detections,
    const std::vector<Detection>& legacy_icons,
    const std::vector<TextRegion>& text_regions,
    const std::vector<TextRecognition>& recognized_text,
    ModelOcrMergeOptions options = {});

} // namespace uiparsercv::pipeline
