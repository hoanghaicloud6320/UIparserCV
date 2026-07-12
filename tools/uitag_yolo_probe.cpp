#include "uiparsercv/infer/onnx_session.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTileSize = 640;
constexpr int kTileStep = 512;
constexpr float kConfidence = 0.25F;
constexpr float kTileNmsIou = 0.70F;  // Ultralytics predict default.
constexpr float kCrossTileNmsIou = 0.50F;  // UITag source default.

constexpr std::array<const char*, 9> kClasses = {
    "Button", "Menu", "Input_Elements", "Navigation",
    "Information_Display", "Sidebar", "Visual_Elements", "Others", "Unknown"};

struct Detection {
  cv::Rect2f box;
  float score{};
  int class_id{};
};

struct Letterbox {
  float scale{};
  int left{};
  int top{};
};

std::vector<float> make_tensor(const cv::Mat& bgr, Letterbox& meta) {
  meta.scale = std::min(
      static_cast<float>(kTileSize) / bgr.cols,
      static_cast<float>(kTileSize) / bgr.rows);
  const int width = static_cast<int>(std::round(bgr.cols * meta.scale));
  const int height = static_cast<int>(std::round(bgr.rows * meta.scale));
  meta.left = (kTileSize - width) / 2;
  meta.top = (kTileSize - height) / 2;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  cv::Mat canvas(kTileSize, kTileSize, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(meta.left, meta.top, width, height)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

  const int plane = kTileSize * kTileSize;
  std::vector<float> tensor(static_cast<std::size_t>(3 * plane));
  for (int y = 0; y < kTileSize; ++y) {
    const auto* row = canvas.ptr<cv::Vec3b>(y);
    for (int x = 0; x < kTileSize; ++x) {
      const int i = y * kTileSize + x;
      tensor[i] = row[x][0] / 255.0F;
      tensor[plane + i] = row[x][1] / 255.0F;
      tensor[2 * plane + i] = row[x][2] / 255.0F;
    }
  }
  return tensor;
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float total = a.area() + b.area() - intersection;
  return total > 0.0F ? intersection / total : 0.0F;
}

std::vector<Detection> nms(std::vector<Detection> detections, float threshold, bool class_aware) {
  std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  std::vector<Detection> result;
  for (const auto& candidate : detections) {
    const bool suppressed = std::any_of(result.begin(), result.end(), [&](const auto& kept) {
      return (!class_aware || kept.class_id == candidate.class_id) &&
          iou(kept.box, candidate.box) > threshold;
    });
    if (!suppressed) result.push_back(candidate);
  }
  return result;
}

std::vector<Detection> infer_tile(
    uiparsercv::infer::OnnxSession& session,
    const cv::Mat& tile,
    int offset_x,
    int offset_y,
    int image_width,
    int image_height) {
  Letterbox meta;
  auto input = make_tensor(tile, meta);
  auto output = session.run(input, {1, 3, kTileSize, kTileSize});
  if (output.shape.size() != 3 || output.shape[1] != 13) {
    throw std::runtime_error("expected UITag output [1,13,N]");
  }

  const int predictions = static_cast<int>(output.shape[2]);
  std::vector<Detection> detections;
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
    if (score < kConfidence) continue;

    const float cx = output.data[i];
    const float cy = output.data[predictions + i];
    const float width = output.data[2 * predictions + i];
    const float height = output.data[3 * predictions + i];
    float x1 = (cx - width / 2.0F - meta.left) / meta.scale + offset_x;
    float y1 = (cy - height / 2.0F - meta.top) / meta.scale + offset_y;
    float x2 = (cx + width / 2.0F - meta.left) / meta.scale + offset_x;
    float y2 = (cy + height / 2.0F - meta.top) / meta.scale + offset_y;
    x1 = std::clamp(x1, static_cast<float>(offset_x), static_cast<float>(offset_x + tile.cols));
    y1 = std::clamp(y1, static_cast<float>(offset_y), static_cast<float>(offset_y + tile.rows));
    x2 = std::clamp(x2, 0.0F, static_cast<float>(image_width));
    y2 = std::clamp(y2, 0.0F, static_cast<float>(image_height));
    if (x2 - x1 >= 1.0F && y2 - y1 >= 1.0F) {
      detections.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score, class_id});
    }
  }
  return nms(std::move(detections), kTileNmsIou, true);
}

std::vector<int> tile_origins(int extent) {
  std::vector<int> origins;
  for (int position = 0;; position += kTileStep) {
    const int end = std::min(position + kTileSize, extent);
    origins.push_back(std::max(0, end - kTileSize));
    if (end >= extent) break;
  }
  origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
  return origins;
}

cv::Scalar color_for(int class_id) {
  static const std::array<cv::Scalar, 9> colors = {
      cv::Scalar(30, 220, 30), cv::Scalar(255, 120, 30), cv::Scalar(30, 200, 255),
      cv::Scalar(220, 60, 220), cv::Scalar(255, 210, 40), cv::Scalar(180, 80, 255),
      cv::Scalar(255, 80, 80), cv::Scalar(100, 220, 220), cv::Scalar(180, 180, 180)};
  return colors[static_cast<std::size_t>(class_id)];
}

void write_json(const std::filesystem::path& path, const cv::Mat& image,
                int tile_count, const std::vector<Detection>& detections) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write JSON: " + path.string());
  out << "{\n  \"image_width\": " << image.cols << ",\n  \"image_height\": " << image.rows
      << ",\n  \"tile_size\": 640,\n  \"tile_step\": 512,\n  \"tile_count\": " << tile_count
      << ",\n  \"confidence_threshold\": 0.25,\n  \"cross_tile_nms_iou\": 0.50,\n  \"detections\": [\n";
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    out << "    {\"class_id\": " << d.class_id << ", \"class\": \"" << kClasses[d.class_id]
        << "\", \"score\": " << std::fixed << std::setprecision(6) << d.score
        << ", \"box_xywh\": [" << d.box.x << ", " << d.box.y << ", "
        << d.box.width << ", " << d.box.height << "]}";
    out << (i + 1 == detections.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2 || argc > 5) {
    std::cerr << "usage: uitag_yolo_probe <image> [overlay.png] [result.json] [model.onnx]\n";
    return 2;
  }
  const std::filesystem::path image_path = argv[1];
  const std::filesystem::path overlay_path = argc >= 3 ? argv[2] : "uitag.overlay.png";
  const std::filesystem::path json_path = argc >= 4 ? argv[3] : "uitag.json";
  const std::filesystem::path model_path = argc >= 5 ? argv[4] : UIPARSERCV_UITAG_MODEL;

  cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) throw std::runtime_error("cannot read image: " + image_path.string());
  uiparsercv::infer::OnnxSession session(model_path);

  std::vector<Detection> all;
  int tile_count = 0;
  for (const int y : tile_origins(image.rows)) {
    for (const int x : tile_origins(image.cols)) {
      const cv::Rect roi(x, y, std::min(kTileSize, image.cols - x), std::min(kTileSize, image.rows - y));
      auto tile_detections = infer_tile(session, image(roi), x, y, image.cols, image.rows);
      all.insert(all.end(), tile_detections.begin(), tile_detections.end());
      ++tile_count;
    }
  }
  // The original UITag implementation intentionally performs class-agnostic
  // cross-tile NMS: overlapping boxes compete even when their labels differ.
  auto detections = nms(std::move(all), kCrossTileNmsIou, false);

  cv::Mat overlay = image.clone();
  for (const auto& d : detections) {
    const auto color = color_for(d.class_id);
    cv::rectangle(overlay, d.box, color, 2, cv::LINE_AA);
    const std::string label = std::string(kClasses[d.class_id]) + " " +
        cv::format("%.2f", d.score);
    int baseline = 0;
    const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
    const int tx = std::max(0, static_cast<int>(d.box.x));
    const int ty = std::max(size.height + 3, static_cast<int>(d.box.y));
    cv::rectangle(overlay, cv::Rect(tx, ty - size.height - 3, size.width + 4, size.height + 4), color, cv::FILLED);
    cv::putText(overlay, label, cv::Point(tx + 2, ty), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(10, 10, 10), 1, cv::LINE_AA);
  }

  if (overlay_path.has_parent_path()) std::filesystem::create_directories(overlay_path.parent_path());
  if (json_path.has_parent_path()) std::filesystem::create_directories(json_path.parent_path());
  if (!cv::imwrite(overlay_path.string(), overlay)) throw std::runtime_error("cannot write overlay");
  write_json(json_path, image, tile_count, detections);

  std::array<int, 9> counts{};
  for (const auto& d : detections) ++counts[static_cast<std::size_t>(d.class_id)];
  std::cout << "model=" << model_path.string() << "\ninput=" << image.cols << "x" << image.rows
            << " tiles=" << tile_count << " detections=" << detections.size() << "\n";
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > 0) std::cout << kClasses[i] << "=" << counts[i] << "\n";
  }
  std::cout << "overlay=" << overlay_path.string() << "\njson=" << json_path.string() << "\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "uitag_yolo_probe: " << error.what() << "\n";
  return 1;
}
