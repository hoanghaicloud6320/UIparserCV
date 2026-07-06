#include "uiparsercv/ocr/ocr_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "clipper.hpp"

namespace uiparsercv::ocr {
namespace {

struct ResizeMeta {
  int original_width{};
  int original_height{};
  int resized_width{};
  int resized_height{};
};

RectF bounding_rect(const std::vector<cv::Point2f>& points) {
  float left = std::numeric_limits<float>::max();
  float top = std::numeric_limits<float>::max();
  float right = std::numeric_limits<float>::lowest();
  float bottom = std::numeric_limits<float>::lowest();

  for (const auto& point : points) {
    left = std::min(left, point.x);
    top = std::min(top, point.y);
    right = std::max(right, point.x);
    bottom = std::max(bottom, point.y);
  }

  return RectF{left, top, right - left, bottom - top};
}

cv::Mat pad_small_image(const cv::Mat& image) {
  if (image.rows + image.cols >= 64) {
    return image;
  }

  cv::Mat padded(std::max(32, image.rows), std::max(32, image.cols), image.type(), cv::Scalar::all(0));
  image.copyTo(padded(cv::Rect(0, 0, image.cols, image.rows)));
  return padded;
}

cv::Mat resize_for_detection(const cv::Mat& image, const OcrDetectorOptions& options, ResizeMeta& meta) {
  const cv::Mat source = pad_small_image(image);
  meta.original_width = image.cols;
  meta.original_height = image.rows;

  int h = source.rows;
  int w = source.cols;
  float ratio = 1.0F;
  // Mirrors PaddleOCR C++ DetResizeForTest::ResizeImageType0.
  if (options.limit_type == "max" && std::max(h, w) > options.limit_side_len) {
    ratio = static_cast<float>(options.limit_side_len) / static_cast<float>(std::max(h, w));
  } else if (options.limit_type == "min" && std::min(h, w) < options.limit_side_len) {
    ratio = static_cast<float>(options.limit_side_len) / static_cast<float>(std::min(h, w));
  } else if (options.limit_type == "resize_long") {
    ratio = static_cast<float>(options.limit_side_len) / static_cast<float>(std::max(h, w));
  } else if (options.limit_type != "max" && options.limit_type != "min") {
    throw std::runtime_error("unsupported OCR detector limit_type: " + options.limit_type);
  }

  int resize_h = static_cast<int>(h * ratio);
  int resize_w = static_cast<int>(w * ratio);
  if (std::max(resize_h, resize_w) > options.max_side_limit) {
    ratio = static_cast<float>(options.max_side_limit) / static_cast<float>(std::max(resize_h, resize_w));
    resize_h = static_cast<int>(resize_h * ratio);
    resize_w = static_cast<int>(resize_w * ratio);
  }

  resize_h = std::max(static_cast<int>(std::round(resize_h / 32.0F) * 32), 32);
  resize_w = std::max(static_cast<int>(std::round(resize_w / 32.0F) * 32), 32);
  meta.resized_width = resize_w;
  meta.resized_height = resize_h;

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
  return resized;
}

std::vector<float> detection_tensor(const cv::Mat& bgr_image, const OcrDetectorOptions& options, ResizeMeta& meta) {
  if (bgr_image.empty()) {
    throw std::runtime_error("OCR detector received an empty image");
  }

  cv::Mat resized = resize_for_detection(bgr_image, options, meta);
  cv::Mat float_image;
  resized.convertTo(float_image, CV_32FC3);

  constexpr float scale = 1.0F / 255.0F;
  const float mean[3] = {0.485F, 0.456F, 0.406F};
  const float stddev[3] = {0.229F, 0.224F, 0.225F};

  const int plane = resized.rows * resized.cols;
  std::vector<float> tensor(static_cast<std::size_t>(3 * plane));
  for (int y = 0; y < resized.rows; ++y) {
    const auto* row = float_image.ptr<cv::Vec3f>(y);
    for (int x = 0; x < resized.cols; ++x) {
      const int idx = y * resized.cols + x;
      for (int c = 0; c < 3; ++c) {
        tensor[c * plane + idx] = (row[x][c] * scale - mean[c]) / stddev[c];
      }
    }
  }

  return tensor;
}

std::pair<std::vector<cv::Point2f>, float> mini_box(const std::vector<cv::Point2f>& contour) {
  cv::RotatedRect rect = cv::minAreaRect(contour);
  std::vector<cv::Point2f> points(4);
  rect.points(points.data());

  std::sort(points.begin(), points.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
    return lhs.x < rhs.x;
  });

  int index_1 = points[1].y > points[0].y ? 0 : 1;
  int index_4 = points[1].y > points[0].y ? 1 : 0;
  int index_2 = points[3].y > points[2].y ? 2 : 3;
  int index_3 = points[3].y > points[2].y ? 3 : 2;

  return {
    {points[index_1], points[index_2], points[index_3], points[index_4]},
    std::min(rect.size.width, rect.size.height)
  };
}

float box_score_fast(const cv::Mat& pred, std::vector<cv::Point2f> contour) {
  // Mirrors PaddleOCR DBPostProcess::BoxScoreFast bounds handling.
  int xmin = std::max(0, static_cast<int>(std::floor(
      std::min_element(contour.begin(), contour.end(), [](const auto& a, const auto& b) {
        return a.x < b.x;
      })->x)));
  int xmax = std::max(0, static_cast<int>(std::ceil(
      std::max_element(contour.begin(), contour.end(), [](const auto& a, const auto& b) {
        return a.x < b.x;
      })->x)));
  int ymin = std::max(0, static_cast<int>(std::floor(
      std::min_element(contour.begin(), contour.end(), [](const auto& a, const auto& b) {
        return a.y < b.y;
      })->y)));
  int ymax = std::max(0, static_cast<int>(std::ceil(
      std::max_element(contour.begin(), contour.end(), [](const auto& a, const auto& b) {
        return a.y < b.y;
      })->y)));

  xmin = std::min(xmin, pred.cols - 1);
  xmax = std::min(xmax, pred.cols - 1);
  ymin = std::min(ymin, pred.rows - 1);
  ymax = std::min(ymax, pred.rows - 1);

  if (xmax < xmin || ymax < ymin) {
    return 0.0F;
  }

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
  std::vector<cv::Point> shifted;
  shifted.reserve(contour.size());
  for (auto& point : contour) {
    shifted.emplace_back(
        static_cast<int>(point.x - xmin),
        static_cast<int>(point.y - ymin));
  }

  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted}, cv::Scalar(1));
  return static_cast<float>(cv::mean(pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask)[0]);
}

std::vector<cv::Point2f> unclip_by_rotated_rect(const std::vector<cv::Point2f>& box, float ratio) {
  const float area = static_cast<float>(std::abs(cv::contourArea(box)));
  const float length = static_cast<float>(cv::arcLength(box, true));
  if (area <= 0.0F || length <= 0.0F) {
    return box;
  }

  const float distance = area * ratio / length;
  ClipperLib::Path path;
  for (const auto& point : box) {
    path << ClipperLib::IntPoint(
        static_cast<ClipperLib::cInt>(std::round(point.x)),
        static_cast<ClipperLib::cInt>(std::round(point.y)));
  }

  ClipperLib::ClipperOffset offset;
  offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths solution;
  offset.Execute(solution, distance);
  if (solution.empty()) {
    return {};
  }

  std::vector<cv::Point2f> result;
  result.reserve(solution[0].size());
  for (const auto& point : solution[0]) {
    result.emplace_back(static_cast<float>(point.X), static_cast<float>(point.Y));
  }
  return result;
}

std::vector<TextRegion> db_postprocess(
    const infer::Tensor& output,
    const ResizeMeta& meta,
    const OcrDetectorOptions& options) {
  if (output.shape.size() != 4 || output.shape[1] != 1) {
    throw std::runtime_error("unexpected OCR detector output shape");
  }

  const int pred_h = static_cast<int>(output.shape[2]);
  const int pred_w = static_cast<int>(output.shape[3]);
  cv::Mat pred(pred_h, pred_w, CV_32FC1, const_cast<float*>(output.data.data()));
  cv::Mat bitmap = pred > options.threshold;

  cv::Mat bitmap_uint8;
  bitmap.convertTo(bitmap_uint8, CV_8UC1, 255.0);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bitmap_uint8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  const int num_contours = std::min(static_cast<int>(contours.size()), options.max_candidates);
  const float width_scale = static_cast<float>(meta.original_width) / static_cast<float>(pred_w);
  const float height_scale = static_cast<float>(meta.original_height) / static_cast<float>(pred_h);

  std::vector<TextRegion> regions;
  for (int i = 0; i < num_contours; ++i) {
    std::vector<cv::Point2f> contour;
    contour.reserve(contours[i].size());
    for (const auto& point : contours[i]) {
      contour.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    auto [box, short_side] = mini_box(contour);
    if (short_side < 3.0F) {
      continue;
    }

    const float score = box_score_fast(pred, box);
    if (score < options.box_threshold) {
      continue;
    }

    std::vector<cv::Point2f> unclipped = unclip_by_rotated_rect(box, options.unclip_ratio);
    auto [scaled_box, scaled_short_side] = mini_box(unclipped);
    if (scaled_short_side < 5.0F) {
      continue;
    }

    for (auto& point : scaled_box) {
      point.x = std::clamp(std::round(point.x * width_scale), 0.0F, static_cast<float>(meta.original_width - 1));
      point.y = std::clamp(std::round(point.y * height_scale), 0.0F, static_cast<float>(meta.original_height - 1));
    }

    regions.push_back(TextRegion{scaled_box, bounding_rect(scaled_box), score});
  }

  std::sort(regions.begin(), regions.end(), [](const TextRegion& lhs, const TextRegion& rhs) {
    if (std::abs(lhs.box.y - rhs.box.y) < 10.0F) {
      return lhs.box.x < rhs.box.x;
    }
    return lhs.box.y < rhs.box.y;
  });

  return regions;
}

} // namespace

OcrTextDetector::OcrTextDetector(OcrDetectorOptions options)
    : options_(std::move(options)),
      session_(options_.model_path) {}

std::vector<TextRegion> OcrTextDetector::detect(const cv::Mat& bgr_image) {
  ResizeMeta meta;
  std::vector<float> input = detection_tensor(bgr_image, options_, meta);
  infer::Tensor output = session_.run(
      input,
      {1, 3, meta.resized_height, meta.resized_width});

  return db_postprocess(output, meta, options_);
}

cv::Mat OcrTextDetector::crop_region(const cv::Mat& bgr_image, const TextRegion& region) const {
  if (region.polygon.size() != 4) {
    throw std::runtime_error("OCR crop requires a quad region");
  }

  const float width_top = cv::norm(region.polygon[0] - region.polygon[1]);
  const float width_bottom = cv::norm(region.polygon[2] - region.polygon[3]);
  const float crop_width = std::max(width_top, width_bottom);

  const float height_left = cv::norm(region.polygon[0] - region.polygon[3]);
  const float height_right = cv::norm(region.polygon[1] - region.polygon[2]);
  const float crop_height = std::max(height_left, height_right);

  std::vector<cv::Point2f> dst{
    {0.0F, 0.0F},
    {crop_width - 1.0F, 0.0F},
    {crop_width - 1.0F, crop_height - 1.0F},
    {0.0F, crop_height - 1.0F}
  };

  // Mirrors PaddleOCR C++ CropByPolys::GetRotateCropImage.
  cv::Mat transform = cv::getPerspectiveTransform(region.polygon, dst);
  cv::Mat crop;
  cv::warpPerspective(
      bgr_image,
      crop,
      transform,
      cv::Size(static_cast<int>(crop_width), static_cast<int>(crop_height)),
      cv::INTER_CUBIC,
      cv::BORDER_REPLICATE);

  if (!crop.empty() && static_cast<float>(crop.rows) / static_cast<float>(crop.cols) >= 1.5F) {
    cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
  }

  return crop;
}

} // namespace uiparsercv::ocr
