#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace uiparsercv::pipeline {
namespace {

tree::Rect to_tree_rect(const RectF& rect) {
  return tree::Rect{
    static_cast<int>(std::round(rect.x)),
    static_cast<int>(std::round(rect.y)),
    std::max(1, static_cast<int>(std::round(rect.width))),
    std::max(1, static_cast<int>(std::round(rect.height)))
  };
}

std::string label_for(const UiElementCandidate& candidate) {
  std::ostringstream label;
  label << (candidate.kind == UiElementKind::Icon ? "icon" : "text");
  if (!candidate.text.empty()) {
    label << ": " << candidate.text;
  } else {
    label << ": " << candidate.source;
  }
  return label.str();
}

tree::TreeNode build_ui_tree(
    const std::vector<UiElementCandidate>& candidates,
    int image_width,
    int image_height) {
  std::vector<tree::Box> boxes;
  boxes.reserve(candidates.size() + 1);

  boxes.push_back(tree::Box{
    1,
    tree::Rect{0, 0, image_width, image_height},
    "screen"
  });

  int id = 2;
  for (const auto& candidate : candidates) {
    boxes.push_back(tree::Box{id++, to_tree_rect(candidate.box), label_for(candidate)});
  }

  return tree::build_containment_tree(std::move(boxes));
}

} // namespace

PipelineRunner::PipelineRunner(PipelineOptions options)
    : options_(std::move(options)),
      icon_detector_(options_.icon),
      ocr_detector_(options_.ocr_detector),
      ocr_recognizer_(options_.ocr_recognizer) {}

PipelineResult PipelineRunner::run(const cv::Mat& bgr_image) {
  if (bgr_image.empty()) {
    throw std::runtime_error("pipeline received an empty image");
  }

  PipelineResult result;
  result.stats.image_width = bgr_image.cols;
  result.stats.image_height = bgr_image.rows;

  result.icons = icon_detector_.detect(bgr_image);
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

  result.candidates = build_candidates(
      result.icons,
      result.text_regions,
      result.recognized_text,
      options_.candidate_merge);

  result.tree = build_ui_tree(
      result.candidates,
      result.stats.image_width,
      result.stats.image_height);

  result.stats.icon_count = result.icons.size();
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

