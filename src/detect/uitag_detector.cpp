#include "uiparsercv/detect/uitag_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace uiparsercv::detect {
namespace {

constexpr std::array<std::string_view, 9> kClasses = {
    "Button", "Menu", "Input_Elements", "Navigation",
    "Information_Display", "Sidebar", "Visual_Elements", "Others", "Unknown"};

float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float total = a.area() + b.area() - intersection;
  return total > 0.0F ? intersection / total : 0.0F;
}

std::vector<float> make_tensor(
    const cv::Mat& bgr, int input_size, UitagLetterbox& meta) {
  meta.scale = std::min(
      static_cast<float>(input_size) / bgr.cols,
      static_cast<float>(input_size) / bgr.rows);
  const int width = static_cast<int>(std::round(bgr.cols * meta.scale));
  const int height = static_cast<int>(std::round(bgr.rows * meta.scale));
  meta.left = (input_size - width) / 2;
  meta.top = (input_size - height) / 2;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(width, height));
  cv::Mat canvas(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(meta.left, meta.top, width, height)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

  const int plane = input_size * input_size;
  std::vector<float> tensor(static_cast<std::size_t>(3 * plane));
  for (int y = 0; y < input_size; ++y) {
    const auto* row = canvas.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input_size; ++x) {
      const int i = y * input_size + x;
      tensor[i] = row[x][0] / 255.0F;
      tensor[plane + i] = row[x][1] / 255.0F;
      tensor[2 * plane + i] = row[x][2] / 255.0F;
    }
  }
  return tensor;
}

}  // namespace

UitagDetector::UitagDetector(
    const std::filesystem::path& model_path, UitagOptions options)
    : session_(model_path), options_(options) {
  if (options_.tile_size <= 0 || options_.tile_step <= 0 ||
      options_.tile_step > options_.tile_size) {
    throw std::invalid_argument("invalid UITag tile options");
  }
}

const UitagOptions& UitagDetector::options() const noexcept { return options_; }

std::vector<int> UitagDetector::tile_origins(
    int extent, int tile_size, int tile_step) {
  if (extent <= 0 || tile_size <= 0 || tile_step <= 0 || tile_step > tile_size) {
    throw std::invalid_argument("invalid tile geometry");
  }
  std::vector<int> origins;
  for (int position = 0;; position += tile_step) {
    const int end = std::min(position + tile_size, extent);
    origins.push_back(std::max(0, end - tile_size));
    if (end >= extent) break;
  }
  origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
  return origins;
}

std::vector<UitagDetection> UitagDetector::decode_output(
    const infer::Tensor& output,
    const UitagLetterbox& letterbox,
    const cv::Size& tile_size,
    const cv::Point& tile_origin,
    const cv::Size& image_size,
    float confidence_threshold) {
  if (output.shape.size() != 3 || output.shape[0] != 1 || output.shape[1] != 13 ||
      letterbox.scale <= 0.0F) {
    throw std::runtime_error("expected UITag output [1,13,N] and valid letterbox");
  }
  const int predictions = static_cast<int>(output.shape[2]);
  if (output.data.size() != static_cast<std::size_t>(13 * predictions)) {
    throw std::runtime_error("UITag output data size does not match shape");
  }

  std::vector<UitagDetection> detections;
  for (int i = 0; i < predictions; ++i) {
    int class_id = 0;
    float score = output.data[4 * predictions + i];
    for (int c = 1; c < 9; ++c) {
      const float value = output.data[(4 + c) * predictions + i];
      if (value > score) {
        score = value;
        class_id = c;
      }
    }
    if (score < confidence_threshold) continue;

    const float cx = output.data[i];
    const float cy = output.data[predictions + i];
    const float width = output.data[2 * predictions + i];
    const float height = output.data[3 * predictions + i];
    float x1 = (cx - width / 2.0F - letterbox.left) / letterbox.scale + tile_origin.x;
    float y1 = (cy - height / 2.0F - letterbox.top) / letterbox.scale + tile_origin.y;
    float x2 = (cx + width / 2.0F - letterbox.left) / letterbox.scale + tile_origin.x;
    float y2 = (cy + height / 2.0F - letterbox.top) / letterbox.scale + tile_origin.y;
    x1 = std::clamp(x1, static_cast<float>(tile_origin.x),
                    static_cast<float>(tile_origin.x + tile_size.width));
    y1 = std::clamp(y1, static_cast<float>(tile_origin.y),
                    static_cast<float>(tile_origin.y + tile_size.height));
    x2 = std::clamp(x2, 0.0F, static_cast<float>(image_size.width));
    y2 = std::clamp(y2, 0.0F, static_cast<float>(image_size.height));
    if (x2 - x1 >= 1.0F && y2 - y1 >= 1.0F) {
      detections.push_back({{x1, y1, x2 - x1, y2 - y1}, score, class_id});
    }
  }
  return detections;
}

std::vector<UitagDetection> UitagDetector::nms(
    std::vector<UitagDetection> detections, float iou_threshold, bool class_aware) {
  std::stable_sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  std::vector<UitagDetection> result;
  for (const auto& candidate : detections) {
    const bool suppressed = std::any_of(result.begin(), result.end(), [&](const auto& kept) {
      return (!class_aware || kept.class_id == candidate.class_id) &&
          intersection_over_union(kept.box, candidate.box) > iou_threshold;
    });
    if (!suppressed) result.push_back(candidate);
  }
  return result;
}

std::string_view UitagDetector::class_name(int class_id) {
  if (class_id < 0 || class_id >= static_cast<int>(kClasses.size())) return "Invalid";
  return kClasses[static_cast<std::size_t>(class_id)];
}

UitagResult UitagDetector::detect(const cv::Mat& bgr_image) {
  if (bgr_image.empty() || bgr_image.type() != CV_8UC3) {
    throw std::invalid_argument("UITag detector expects a non-empty CV_8UC3 image");
  }
  UitagResult result;
  for (const int y : tile_origins(bgr_image.rows, options_.tile_size, options_.tile_step)) {
    for (const int x : tile_origins(bgr_image.cols, options_.tile_size, options_.tile_step)) {
      const cv::Rect roi(
          x, y, std::min(options_.tile_size, bgr_image.cols - x),
          std::min(options_.tile_size, bgr_image.rows - y));
      UitagLetterbox letterbox;
      auto input = make_tensor(bgr_image(roi), options_.tile_size, letterbox);
      const auto output = session_.run(
          input, {1, 3, options_.tile_size, options_.tile_size});
      auto tile_detections = decode_output(
          output, letterbox, roi.size(), roi.tl(), bgr_image.size(),
          options_.confidence_threshold);
      tile_detections = nms(
          std::move(tile_detections), options_.tile_nms_iou, true);
      result.raw_detections.insert(
          result.raw_detections.end(), tile_detections.begin(), tile_detections.end());
      ++result.tile_count;
    }
  }
  result.detections = nms(
      result.raw_detections, options_.cross_tile_nms_iou, false);
  return result;
}

}  // namespace uiparsercv::detect
