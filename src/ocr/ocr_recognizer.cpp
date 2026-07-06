#include "uiparsercv/ocr/ocr_recognizer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace uiparsercv::ocr {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string parse_yaml_scalar(std::string value) {
  value = trim(std::move(value));
  if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                            (value.front() == '"' && value.back() == '"'))) {
    value = value.substr(1, value.size() - 2);
  }
  if (value == "\\\\") {
    return "\\";
  }
  return value;
}

std::vector<std::string> load_character_dict(const std::filesystem::path& config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("failed to open OCR recognizer config: " + config_path.string());
  }

  std::vector<std::string> characters;
  bool in_character_dict = false;
  std::string line;
  while (std::getline(input, line)) {
    const std::string stripped = trim(line);
    if (stripped == "character_dict:") {
      in_character_dict = true;
      continue;
    }

    if (!in_character_dict) {
      continue;
    }

    if (stripped.rfind("- ", 0) == 0) {
      characters.push_back(parse_yaml_scalar(stripped.substr(2)));
      continue;
    }

    if (!stripped.empty() && line.rfind("  ", 0) != 0) {
      break;
    }
  }

  characters.insert(characters.begin(), "blank");
  return characters;
}

std::vector<float> recognition_tensor(
    const cv::Mat& bgr_crop,
    const OcrRecognizerOptions& options,
    std::vector<int64_t>& shape) {
  if (bgr_crop.empty()) {
    throw std::runtime_error("OCR recognizer received an empty crop");
  }

  const float rec_wh_ratio = static_cast<float>(options.image_width) / static_cast<float>(options.image_height);
  const float image_wh_ratio = static_cast<float>(bgr_crop.cols) / static_cast<float>(bgr_crop.rows);
  const float max_wh_ratio = std::max(rec_wh_ratio, image_wh_ratio);

  // Mirrors PaddleOCR C++ OCRReisizeNormImg::ResizeNormImg.
  int target_width = static_cast<int>(std::ceil(options.image_height * max_wh_ratio));
  target_width = std::min(target_width, options.max_width);
  int resized_width = std::min(
      target_width,
      static_cast<int>(std::ceil(options.image_height * image_wh_ratio)));
  resized_width = std::max(1, resized_width);

  cv::Mat resized;
  cv::resize(bgr_crop, resized, cv::Size(resized_width, options.image_height), 0, 0, cv::INTER_LINEAR);

  cv::Mat float_image;
  resized.convertTo(float_image, CV_32FC3);

  const int plane = options.image_height * target_width;
  std::vector<float> tensor(static_cast<std::size_t>(options.image_channels * plane), 0.0F);

  for (int y = 0; y < options.image_height; ++y) {
    const auto* row = float_image.ptr<cv::Vec3f>(y);
    for (int x = 0; x < resized_width; ++x) {
      const int idx = y * target_width + x;
      for (int c = 0; c < options.image_channels; ++c) {
        tensor[c * plane + idx] = (row[x][c] / 255.0F - 0.5F) / 0.5F;
      }
    }
  }

  shape = {1, options.image_channels, options.image_height, target_width};
  return tensor;
}

TextRecognition ctc_decode(const infer::Tensor& output, const std::vector<std::string>& characters) {
  if (output.shape.size() != 3) {
    throw std::runtime_error("unexpected OCR recognizer output shape");
  }

  const int seq_len = static_cast<int>(output.shape[1]);
  const int classes = static_cast<int>(output.shape[2]);
  std::vector<int> indices;
  std::vector<float> probabilities;
  indices.reserve(static_cast<std::size_t>(seq_len));
  probabilities.reserve(static_cast<std::size_t>(seq_len));

  for (int t = 0; t < seq_len; ++t) {
    const float* row = output.data.data() + static_cast<std::size_t>(t * classes);
    int best_index = 0;
    float best_score = row[0];
    for (int c = 1; c < classes; ++c) {
      if (row[c] > best_score) {
        best_score = row[c];
        best_index = c;
      }
    }
    indices.push_back(best_index);
    probabilities.push_back(best_score);
  }

  std::string text;
  std::vector<float> kept_scores;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] == 0) {
      continue;
    }
    if (i > 0 && indices[i] == indices[i - 1]) {
      continue;
    }
    if (indices[i] < static_cast<int>(characters.size())) {
      text += characters[static_cast<std::size_t>(indices[i])];
      kept_scores.push_back(probabilities[i]);
    }
  }

  const float confidence = kept_scores.empty()
      ? 0.0F
      : std::accumulate(kept_scores.begin(), kept_scores.end(), 0.0F) /
            static_cast<float>(kept_scores.size());
  return TextRecognition{text, confidence};
}

} // namespace

OcrTextRecognizer::OcrTextRecognizer(OcrRecognizerOptions options)
    : options_(std::move(options)),
      session_(options_.model_path),
      characters_(load_character_dict(options_.config_path)) {}

TextRecognition OcrTextRecognizer::recognize(const cv::Mat& bgr_crop) {
  std::vector<int64_t> shape;
  std::vector<float> input = recognition_tensor(bgr_crop, options_, shape);
  infer::Tensor output = session_.run(input, shape);
  return ctc_decode(output, characters_);
}

const std::vector<std::string>& OcrTextRecognizer::character_list() const {
  return characters_;
}

} // namespace uiparsercv::ocr
