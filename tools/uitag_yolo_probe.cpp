#include "uiparsercv/detect/uitag_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

cv::Scalar color_for(int class_id) {
  static const std::array<cv::Scalar, 9> colors = {
      cv::Scalar(30, 220, 30), cv::Scalar(255, 120, 30), cv::Scalar(30, 200, 255),
      cv::Scalar(220, 60, 220), cv::Scalar(255, 210, 40), cv::Scalar(180, 80, 255),
      cv::Scalar(255, 80, 80), cv::Scalar(100, 220, 220), cv::Scalar(180, 180, 180)};
  return colors[static_cast<std::size_t>(class_id)];
}

void write_json(
    const std::filesystem::path& path,
    const cv::Mat& image,
    const std::filesystem::path& model_path,
    const uiparsercv::detect::UitagOptions& options,
    const uiparsercv::detect::UitagResult& result) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write JSON: " + path.string());
  out << "{\n  \"model\": \"" << model_path.generic_string()
      << "\",\n  \"image_width\": " << image.cols
      << ",\n  \"image_height\": " << image.rows
      << ",\n  \"tile_size\": " << options.tile_size
      << ",\n  \"tile_step\": " << options.tile_step
      << ",\n  \"tile_count\": " << result.tile_count
      << ",\n  \"confidence_threshold\": " << options.confidence_threshold
      << ",\n  \"tile_nms_iou\": " << options.tile_nms_iou
      << ",\n  \"cross_tile_nms_iou\": " << options.cross_tile_nms_iou
      << ",\n  \"raw_detection_count\": " << result.raw_detections.size()
      << ",\n  \"detections\": [\n";
  for (std::size_t i = 0; i < result.detections.size(); ++i) {
    const auto& detection = result.detections[i];
    out << "    {\"class_id\": " << detection.class_id << ", \"class\": \""
        << uiparsercv::detect::UitagDetector::class_name(detection.class_id)
        << "\", \"score\": " << std::fixed << std::setprecision(6) << detection.score
        << ", \"box_xywh\": [" << detection.box.x << ", " << detection.box.y << ", "
        << detection.box.width << ", " << detection.box.height << "]}"
        << (i + 1 == result.detections.size() ? "\n" : ",\n");
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

  uiparsercv::detect::UitagDetector detector(model_path);
  const auto result = detector.detect(image);

  cv::Mat overlay = image.clone();
  for (const auto& detection : result.detections) {
    const auto color = color_for(detection.class_id);
    cv::rectangle(overlay, detection.box, color, 2, cv::LINE_AA);
    const std::string label =
        std::string(uiparsercv::detect::UitagDetector::class_name(detection.class_id)) +
        " " + cv::format("%.2f", detection.score);
    int baseline = 0;
    const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
    const int tx = std::max(0, static_cast<int>(detection.box.x));
    const int ty = std::max(size.height + 3, static_cast<int>(detection.box.y));
    cv::rectangle(overlay, {tx, ty - size.height - 3, size.width + 4, size.height + 4},
                  color, cv::FILLED);
    cv::putText(overlay, label, {tx + 2, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(10, 10, 10), 1, cv::LINE_AA);
  }

  if (overlay_path.has_parent_path()) std::filesystem::create_directories(overlay_path.parent_path());
  if (json_path.has_parent_path()) std::filesystem::create_directories(json_path.parent_path());
  if (!cv::imwrite(overlay_path.string(), overlay)) throw std::runtime_error("cannot write overlay");
  write_json(json_path, image, model_path, detector.options(), result);

  std::array<int, 9> counts{};
  for (const auto& detection : result.detections) {
    ++counts[static_cast<std::size_t>(detection.class_id)];
  }
  std::cout << "model=" << model_path.string() << "\ninput=" << image.cols << 'x' << image.rows
            << " tiles=" << result.tile_count << " raw=" << result.raw_detections.size()
            << " detections=" << result.detections.size() << '\n';
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > 0) {
      std::cout << uiparsercv::detect::UitagDetector::class_name(static_cast<int>(i))
                << '=' << counts[i] << '\n';
    }
  }
  std::cout << "overlay=" << overlay_path.string() << "\njson=" << json_path.string() << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "uitag_yolo_probe: " << error.what() << '\n';
  return 1;
}
