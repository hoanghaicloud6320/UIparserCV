#include "uiparsercv/pipeline/visual_container_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace uiparsercv::pipeline {
namespace {

float rect_area(const RectF& rect) {
  return std::max(0.0F, rect.width) * std::max(0.0F, rect.height);
}

float iou(const RectF& lhs, const RectF& rhs) {
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  return intersection / (rect_area(lhs) + rect_area(rhs) - intersection + 1e-6F);
}

RectF to_rectf(const cv::Rect& rect) {
  return RectF{
    static_cast<float>(rect.x),
    static_cast<float>(rect.y),
    static_cast<float>(rect.width),
    static_cast<float>(rect.height)
  };
}

bool is_duplicate(
    const RectF& box,
    const std::vector<UiElementCandidate>& existing,
    const std::vector<UiElementCandidate>& detected,
    float threshold) {
  const auto overlaps = [&](const UiElementCandidate& candidate) {
    return iou(box, candidate.box) >= threshold;
  };
  return std::any_of(existing.begin(), existing.end(), overlaps) ||
         std::any_of(detected.begin(), detected.end(), overlaps);
}

bool merely_wraps_one_candidate(
    const RectF& box,
    const std::vector<UiElementCandidate>& existing) {
  const float box_area = rect_area(box);
  for (const auto& candidate : existing) {
    const float candidate_area = rect_area(candidate.box);
    if (candidate_area <= 0.0F || box_area > candidate_area * 1.45F) {
      continue;
    }
    const float left = std::max(box.x, candidate.box.x);
    const float top = std::max(box.y, candidate.box.y);
    const float right = std::min(box.x + box.width, candidate.box.x + candidate.box.width);
    const float bottom = std::min(box.y + box.height, candidate.box.y + candidate.box.height);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    if (intersection / candidate_area >= 0.90F) {
      return true;
    }
  }
  return false;
}

double edge_strength(const cv::Mat& gray, const cv::Rect& rect) {
  const int inset = std::clamp(std::min(rect.width, rect.height) / 20, 1, 5);
  if (rect.x < inset || rect.y < inset ||
      rect.x + rect.width + inset >= gray.cols ||
      rect.y + rect.height + inset >= gray.rows) {
    return 0.0;
  }

  const int x0 = rect.x + inset;
  const int x1 = rect.x + rect.width - inset - 1;
  const int y0 = rect.y + inset;
  const int y1 = rect.y + rect.height - inset - 1;
  if (x1 <= x0 || y1 <= y0) {
    return 0.0;
  }

  double sum = 0.0;
  int count = 0;
  const int step_x = std::max(1, rect.width / 32);
  const int step_y = std::max(1, rect.height / 32);
  for (int x = x0; x <= x1; x += step_x) {
    sum += std::abs(gray.at<unsigned char>(y0, x) - gray.at<unsigned char>(y0 - inset, x));
    sum += std::abs(gray.at<unsigned char>(y1, x) - gray.at<unsigned char>(y1 + inset, x));
    count += 2;
  }
  for (int y = y0; y <= y1; y += step_y) {
    sum += std::abs(gray.at<unsigned char>(y, x0) - gray.at<unsigned char>(y, x0 - inset));
    sum += std::abs(gray.at<unsigned char>(y, x1) - gray.at<unsigned char>(y, x1 + inset));
    count += 2;
  }
  return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

} // namespace

std::vector<UiElementCandidate> detect_visual_containers(
    const cv::Mat& bgr_image,
    const std::vector<UiElementCandidate>& existing,
    VisualContainerOptions options) {
  std::vector<UiElementCandidate> result;
  if (!options.enabled || bgr_image.empty()) {
    return result;
  }

  cv::Mat smooth;
  cv::bilateralFilter(bgr_image, smooth, 7, 35.0, 35.0);
  cv::Mat gray;
  cv::cvtColor(smooth, gray, cv::COLOR_BGR2GRAY);
  cv::Mat edges;
  cv::Canny(gray, edges, 28.0, 84.0);
  cv::morphologyEx(
      edges,
      edges,
      cv::MORPH_CLOSE,
      cv::getStructuringElement(cv::MORPH_RECT, {5, 5}),
      {-1, -1},
      2);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  const float image_area = static_cast<float>(bgr_image.cols) * static_cast<float>(bgr_image.rows);
  for (const auto& contour : contours) {
    const cv::Rect rect = cv::boundingRect(contour);
    if (rect.width < options.min_side || rect.height < options.min_side) {
      continue;
    }

    const float area_ratio = static_cast<float>(rect.area()) / image_area;
    if (area_ratio < options.min_area_ratio || area_ratio > options.max_area_ratio) {
      continue;
    }

    const float contour_area = static_cast<float>(std::abs(cv::contourArea(contour)));
    const float rectangularity = contour_area / static_cast<float>(std::max(1, rect.area()));
    if (rectangularity < options.min_rectangularity) {
      continue;
    }

    const double boundary_contrast = edge_strength(gray, rect);
    if (boundary_contrast < 5.0) {
      continue;
    }

    const RectF box = to_rectf(rect);
    if (is_duplicate(box, existing, result, options.duplicate_iou) ||
        merely_wraps_one_candidate(box, existing)) {
      continue;
    }

    const float score = std::clamp(
        rectangularity * 0.75F + static_cast<float>(boundary_contrast / 80.0) * 0.25F,
        0.0F,
        1.0F);
    result.push_back(UiElementCandidate{
      UiElementKind::VisualContainer,
      box,
      score,
      "",
      "",
      "",
      0.0F,
      false,
      "visual_color_container"
    });
  }

  std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return rect_area(lhs.box) > rect_area(rhs.box);
  });
  return result;
}

} // namespace uiparsercv::pipeline
