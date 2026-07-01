#pragma once

#include "uiparsercv/common/types.hpp"
#include "uiparsercv/icon/icon_detector.hpp"
#include "uiparsercv/ocr/ocr_detector.hpp"
#include "uiparsercv/ocr/ocr_recognizer.hpp"
#include "uiparsercv/pipeline/ui_element.hpp"
#include "uiparsercv/tree/box_tree.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <vector>

namespace uiparsercv::pipeline {

struct PipelineOptions {
  icon::IconDetectorOptions icon;
  ocr::OcrDetectorOptions ocr_detector;
  ocr::OcrRecognizerOptions ocr_recognizer;
  CandidateMergeOptions candidate_merge;
};

struct PipelineStats {
  int image_width{};
  int image_height{};
  std::size_t icon_count{};
  std::size_t text_region_count{};
  std::size_t candidate_count{};
};

struct PipelineResult {
  PipelineStats stats;
  std::vector<Detection> icons;
  std::vector<TextRegion> text_regions;
  std::vector<TextRecognition> recognized_text;
  std::vector<UiElementCandidate> candidates;
  tree::TreeNode tree;
};

class PipelineRunner {
public:
  explicit PipelineRunner(PipelineOptions options = {});

  [[nodiscard]] PipelineResult run(const cv::Mat& bgr_image);
  [[nodiscard]] PipelineResult run_file(const std::filesystem::path& image_path);

private:
  PipelineOptions options_;
  icon::IconDetector icon_detector_;
  ocr::OcrTextDetector ocr_detector_;
  ocr::OcrTextRecognizer ocr_recognizer_;
};

} // namespace uiparsercv::pipeline

