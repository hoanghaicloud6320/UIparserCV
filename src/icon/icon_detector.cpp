#include "uiparsercv/icon/icon_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uiparsercv::icon {
namespace {

struct LetterboxMeta {
  float scale{};
  float pad_x{};
  float pad_y{};
};

float intersection_over_union(const RectF& lhs, const RectF& rhs) {
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float w = std::max(0.0F, right - left);
  const float h = std::max(0.0F, bottom - top);
  const float inter = w * h;
  const float area_lhs = std::max(0.0F, lhs.width) * std::max(0.0F, lhs.height);
  const float area_rhs = std::max(0.0F, rhs.width) * std::max(0.0F, rhs.height);
  const float denom = area_lhs + area_rhs - inter;
  return denom <= 0.0F ? 0.0F : inter / denom;
}

std::vector<float> letterbox_to_tensor(
    const cv::Mat& bgr_image,
    int target_size,
    LetterboxMeta& meta) {
  if (bgr_image.empty()) {
    throw std::runtime_error("icon detector received an empty image");
  }

  meta.scale = std::min(
      static_cast<float>(target_size) / static_cast<float>(bgr_image.cols),
      static_cast<float>(target_size) / static_cast<float>(bgr_image.rows));

  const int resized_w = static_cast<int>(std::round(bgr_image.cols * meta.scale));
  const int resized_h = static_cast<int>(std::round(bgr_image.rows * meta.scale));
  meta.pad_x = static_cast<float>(target_size - resized_w) / 2.0F;
  meta.pad_y = static_cast<float>(target_size - resized_h) / 2.0F;

  cv::Mat resized;
  cv::resize(bgr_image, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat canvas(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(
      static_cast<int>(std::round(meta.pad_x)),
      static_cast<int>(std::round(meta.pad_y)),
      resized_w,
      resized_h)));

  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

  std::vector<float> tensor(static_cast<std::size_t>(3 * target_size * target_size));
  const int plane = target_size * target_size;
  for (int y = 0; y < target_size; ++y) {
    const auto* row = rgb.ptr<cv::Vec3b>(y);
    for (int x = 0; x < target_size; ++x) {
      const int idx = y * target_size + x;
      tensor[idx] = static_cast<float>(row[x][0]) / 255.0F;
      tensor[plane + idx] = static_cast<float>(row[x][1]) / 255.0F;
      tensor[2 * plane + idx] = static_cast<float>(row[x][2]) / 255.0F;
    }
  }

  return tensor;
}

RectF clamp_box(RectF box, int width, int height) {
  const float x1 = std::clamp(box.x, 0.0F, static_cast<float>(width - 1));
  const float y1 = std::clamp(box.y, 0.0F, static_cast<float>(height - 1));
  const float x2 = std::clamp(box.x + box.width, 0.0F, static_cast<float>(width - 1));
  const float y2 = std::clamp(box.y + box.height, 0.0F, static_cast<float>(height - 1));
  return RectF{x1, y1, std::max(0.0F, x2 - x1), std::max(0.0F, y2 - y1)};
}

std::vector<Detection> nms(std::vector<Detection> detections, float threshold) {
  std::sort(detections.begin(), detections.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.score > rhs.score;
  });

  std::vector<Detection> kept;
  std::vector<bool> removed(detections.size(), false);
  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    for (std::size_t j = i + 1; j < detections.size(); ++j) {
      if (!removed[j] && intersection_over_union(detections[i].box, detections[j].box) > threshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

} // namespace

IconDetector::IconDetector(IconDetectorOptions options)
    : options_(std::move(options)),
      session_(options_.model_path) {}

std::vector<Detection> IconDetector::detect(const cv::Mat& bgr_image) {
  LetterboxMeta meta;
  std::vector<float> input = letterbox_to_tensor(bgr_image, options_.input_size, meta);
  infer::Tensor output = session_.run(
      input,
      {1, 3, options_.input_size, options_.input_size});

  if (output.shape.size() != 3 || output.shape[1] != 5) {
    throw std::runtime_error("unexpected icon detector output shape");
  }

  const int anchors = static_cast<int>(output.shape[2]);
  std::vector<Detection> detections;
  detections.reserve(static_cast<std::size_t>(anchors));

  for (int i = 0; i < anchors; ++i) {
    const float score = output.data[4 * anchors + i];
    if (score < options_.confidence_threshold) {
      continue;
    }

    const float cx = output.data[i];
    const float cy = output.data[anchors + i];
    const float w = output.data[2 * anchors + i];
    const float h = output.data[3 * anchors + i];

    RectF box{
      (cx - w / 2.0F - meta.pad_x) / meta.scale,
      (cy - h / 2.0F - meta.pad_y) / meta.scale,
      w / meta.scale,
      h / meta.scale
    };

    detections.push_back(Detection{
      clamp_box(box, bgr_image.cols, bgr_image.rows),
      score,
      "icon"
    });
  }

  return nms(std::move(detections), options_.nms_threshold);
}

} // namespace uiparsercv::icon

