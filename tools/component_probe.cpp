#include "uiparsercv/icon/icon_detector.hpp"
#include "uiparsercv/ocr/ocr_detector.hpp"
#include "uiparsercv/ocr/ocr_recognizer.hpp"
#include "uiparsercv/pipeline/ui_element.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>

namespace {

cv::Mat make_probe_image() {
  cv::Mat image(220, 640, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(image, cv::Rect(24, 24, 592, 172), cv::Scalar(230, 230, 230), 2);
  cv::putText(
      image,
      "UIparserCV OCR smoke test",
      cv::Point(44, 112),
      cv::FONT_HERSHEY_SIMPLEX,
      1.0,
      cv::Scalar(20, 20, 20),
      2,
      cv::LINE_AA);
  return image;
}

cv::Mat load_or_make_image(int argc, char** argv) {
  if (argc == 1) {
    return make_probe_image();
  }

  if (argc != 2) {
    throw std::runtime_error("usage: component_probe [image-path]");
  }

  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error(std::string("failed to load image: ") + argv[1]);
  }
  return image;
}

} // namespace

int main(int argc, char** argv) {
  try {
    cv::Mat image = load_or_make_image(argc, argv);

    uiparsercv::icon::IconDetector icons({
      UIPARSERCV_ICON_MODEL,
      640,
      0.05F,
      0.1F
    });

    uiparsercv::ocr::OcrDetectorOptions detector_options;
    detector_options.model_path = UIPARSERCV_OCR_DET_MODEL;
    detector_options.limit_side_len = 736;
    detector_options.limit_type = "min";
    detector_options.max_side_limit = 4000;
    detector_options.threshold = 0.2F;
    detector_options.box_threshold = 0.4F;
    detector_options.unclip_ratio = 1.4F;
    detector_options.max_candidates = 3000;
    uiparsercv::ocr::OcrTextDetector detector(detector_options);

    uiparsercv::ocr::OcrTextRecognizer recognizer({
      UIPARSERCV_OCR_REC_MODEL,
      UIPARSERCV_OCR_REC_CONFIG,
      3,
      48,
      320,
      3200
    });

    const auto icon_results = icons.detect(image);
    const auto text_regions = detector.detect(image);
    std::vector<uiparsercv::TextRecognition> text_results;
    text_results.reserve(text_regions.size());

    std::cout << "image: " << image.cols << "x" << image.rows << '\n';
    std::cout << "icons: " << icon_results.size() << '\n';
    for (std::size_t i = 0; i < std::min<std::size_t>(icon_results.size(), 5); ++i) {
      const auto& icon = icon_results[i];
      std::cout << "  icon[" << i << "] score=" << icon.score
                << " box=(" << icon.box.x << ", " << icon.box.y << ", "
                << icon.box.width << ", " << icon.box.height << ")\n";
    }

    std::cout << "text_regions: " << text_regions.size() << '\n';
    for (std::size_t i = 0; i < std::min<std::size_t>(text_regions.size(), 5); ++i) {
      const auto& region = text_regions[i];
      cv::Mat crop = detector.crop_region(image, region);
      if (crop.empty()) {
        continue;
      }
      const auto text = recognizer.recognize(crop);
      text_results.push_back(text);
      std::cout << "  text[" << i << "] score=" << region.score
                << " rec_conf=" << text.confidence
                << " value=\"" << text.text << "\"\n";
    }

    if (text_results.size() < text_regions.size()) {
      text_results.resize(text_regions.size());
    }

    const auto candidates = uiparsercv::pipeline::build_candidates(
        icon_results,
        text_regions,
        text_results,
        {0.1F});
    std::cout << "candidates: " << candidates.size() << '\n';
    for (std::size_t i = 0; i < std::min<std::size_t>(candidates.size(), 8); ++i) {
      const auto& candidate = candidates[i];
      std::cout << "  candidate[" << i << "] kind="
                << (candidate.kind == uiparsercv::pipeline::UiElementKind::Icon
                        ? "icon"
                        : (candidate.kind == uiparsercv::pipeline::UiElementKind::Text
                               ? "text"
                               : (candidate.kind == uiparsercv::pipeline::UiElementKind::VisualContainer
                                      ? "container"
                                      : "group")))
                << " source=" << candidate.source
                << " score=" << candidate.detection_score
                << " text=\"" << candidate.text << "\"\n";
    }

    if (text_regions.empty()) {
      cv::Rect fallback_rect(0, 0, image.cols, image.rows);
      cv::Mat fallback = image(fallback_rect).clone();
      const auto text = recognizer.recognize(fallback);
      std::cout << "fallback_recognition: conf=" << text.confidence
                << " value=\"" << text.text << "\"\n";
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "component_probe error: " << error.what() << '\n';
    return 1;
  }
}
