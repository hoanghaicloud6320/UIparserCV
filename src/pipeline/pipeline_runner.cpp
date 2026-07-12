#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include "uiparsercv/pipeline/tree_grouper.hpp"
#include "uiparsercv/pipeline/boundaryless_group_detector.hpp"
#include "uiparsercv/pipeline/line_rect_detector.hpp"
#include "uiparsercv/pipeline/visual_container_detector.hpp"

#include <opencv2/imgcodecs.hpp>

#include <stdexcept>
#include <utility>

namespace uiparsercv::pipeline {

PipelineRunner::PipelineRunner(PipelineOptions options)
    : options_(std::move(options)),
      uitag_detector_(options_.uitag_model_path, options_.uitag),
      ocr_detector_(options_.ocr_detector),
      ocr_recognizer_(options_.ocr_recognizer) {
  if (options_.enable_legacy_icon_support) {
    icon_detector_ = std::make_unique<icon::IconDetector>(options_.icon);
  }
}

PipelineResult PipelineRunner::run(const cv::Mat& bgr_image) {
  if (bgr_image.empty()) {
    throw std::runtime_error("pipeline received an empty image");
  }

  PipelineResult result;
  result.stats.image_width = bgr_image.cols;
  result.stats.image_height = bgr_image.rows;

  const auto uitag_result = uitag_detector_.detect(bgr_image);
  result.uitag_raw_detections = uitag_result.raw_detections;
  result.uitag_detections = uitag_result.detections;
  if (icon_detector_) {
    result.icons = icon_detector_->detect(bgr_image);
  }
  result.text_regions = ocr_detector_.detect(bgr_image);
  result.recognized_text.reserve(result.text_regions.size());

  for (const auto& region : result.text_regions) {
    cv::Mat crop = ocr_detector_.crop_region(bgr_image, region);
    if (crop.empty()) {
      result.recognized_text.push_back(TextRecognition{});
      continue;
    }
    result.recognized_text.push_back(ocr_recognizer_.recognize(crop));
  }

  auto combined = build_model_ocr_candidates(
      result.uitag_detections,
      result.icons,
      result.text_regions,
      result.recognized_text,
      options_.model_ocr_merge);
  result.candidates = std::move(combined.candidates);
  result.associations = std::move(combined.associations);

  if (options_.enable_legacy_heuristics) {
    auto visual_containers = detect_visual_containers(
        bgr_image, result.candidates, options_.visual_containers);
    result.stats.visual_container_count = visual_containers.size();
    result.candidates.insert(
        result.candidates.end(), visual_containers.begin(), visual_containers.end());

    auto line_rects = detect_line_rects(
        bgr_image, result.candidates, options_.line_rects);
    result.stats.line_rect_count = line_rects.size();
    result.candidates.insert(
        result.candidates.end(), line_rects.begin(), line_rects.end());

    auto inferred_groups = detect_boundaryless_groups(
        result.candidates, result.stats.image_width, result.stats.image_height,
        options_.boundaryless_groups);
    result.stats.inferred_group_count = inferred_groups.size();
    result.candidates.insert(
        result.candidates.end(), inferred_groups.begin(), inferred_groups.end());
  }

  result.tree = build_grouped_ui_tree(
      result.candidates,
      result.stats.image_width,
      result.stats.image_height);

  result.stats.icon_count = result.icons.size();
  result.stats.uitag_raw_count = result.uitag_raw_detections.size();
  result.stats.uitag_count = result.uitag_detections.size();
  result.stats.association_count = result.associations.size();
  result.stats.text_region_count = result.text_regions.size();
  result.stats.candidate_count = result.candidates.size();

  return result;
}

PipelineResult PipelineRunner::run_file(const std::filesystem::path& image_path) {
  cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("failed to load image: " + image_path.string());
  }
  return run(image);
}

} // namespace uiparsercv::pipeline
