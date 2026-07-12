#pragma once

#include "uiparsercv/common/types.hpp"
#include "uiparsercv/detect/uitag_detector.hpp"
#include "uiparsercv/icon/icon_detector.hpp"
#include "uiparsercv/ocr/ocr_detector.hpp"
#include "uiparsercv/ocr/ocr_recognizer.hpp"
#include "uiparsercv/pipeline/ui_element.hpp"
#include "uiparsercv/pipeline/boundaryless_group_detector.hpp"
#include "uiparsercv/pipeline/line_rect_detector.hpp"
#include "uiparsercv/pipeline/visual_container_detector.hpp"
#include "uiparsercv/tree/box_tree.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace uiparsercv::pipeline {

struct PipelineOptions {
  std::filesystem::path uitag_model_path{"models/uitag/uitag-yolo11s-ui-detect-v1.onnx"};
  detect::UitagOptions uitag;
  bool enable_legacy_icon_support{true};
  bool enable_legacy_heuristics{false};
  icon::IconDetectorOptions icon;
  ocr::OcrDetectorOptions ocr_detector;
  ocr::OcrRecognizerOptions ocr_recognizer;
  CandidateMergeOptions candidate_merge;
  ModelOcrMergeOptions model_ocr_merge;
  VisualContainerOptions visual_containers;
  BoundarylessGroupOptions boundaryless_groups;
  LineRectOptions line_rects;
};

struct PipelineStats {
  int image_width{};
  int image_height{};
  std::size_t icon_count{};
  std::size_t uitag_raw_count{};
  std::size_t uitag_count{};
  std::size_t association_count{};
  std::size_t text_region_count{};
  std::size_t candidate_count{};
  std::size_t visual_container_count{};
  std::size_t inferred_group_count{};
  std::size_t line_rect_count{};
};

struct PipelineResult {
  PipelineStats stats;
  std::vector<Detection> icons;
  std::vector<detect::UitagDetection> uitag_raw_detections;
  std::vector<detect::UitagDetection> uitag_detections;
  std::vector<TextRegion> text_regions;
  std::vector<TextRecognition> recognized_text;
  std::vector<UiElementCandidate> candidates;
  std::vector<CandidateAssociation> associations;
  tree::TreeNode tree;
};

class PipelineRunner {
public:
  explicit PipelineRunner(PipelineOptions options = {});

  [[nodiscard]] PipelineResult run(const cv::Mat& bgr_image);
  [[nodiscard]] PipelineResult run_file(const std::filesystem::path& image_path);

private:
  PipelineOptions options_;
  detect::UitagDetector uitag_detector_;
  std::unique_ptr<icon::IconDetector> icon_detector_;
  ocr::OcrTextDetector ocr_detector_;
  ocr::OcrTextRecognizer ocr_recognizer_;
};

} // namespace uiparsercv::pipeline
