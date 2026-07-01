#include "uiparsercv/export/debug_overlay.hpp"
#include "uiparsercv/export/json_exporter.hpp"
#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Combo {
  std::string name;
  std::filesystem::path det_model;
  std::filesystem::path rec_model;
  std::filesystem::path rec_config;
};

void write_json_file(
    const std::filesystem::path& path,
    const uiparsercv::pipeline::PipelineResult& result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  uiparsercv::exporter::write_json(out, result);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: model_combo_compare <input-image> [output-dir]\n";
      return 1;
    }

    const std::filesystem::path input = argv[1];
    const std::filesystem::path output_dir =
        argc == 3 ? std::filesystem::path(argv[2]) : std::filesystem::path("model_combo_compare");
    std::filesystem::create_directories(output_dir);

    const cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to load image: " + input.string());
    }

    const std::vector<Combo> combos{
      {
        "tiny_det_tiny_rec",
        UIPARSERCV_OCR_TINY_DET_MODEL,
        UIPARSERCV_OCR_TINY_REC_MODEL,
        UIPARSERCV_OCR_TINY_REC_CONFIG
      },
      {
        "tiny_det_small_rec",
        UIPARSERCV_OCR_TINY_DET_MODEL,
        UIPARSERCV_OCR_SMALL_REC_MODEL,
        UIPARSERCV_OCR_SMALL_REC_CONFIG
      },
      {
        "small_det_small_rec",
        UIPARSERCV_OCR_SMALL_DET_MODEL,
        UIPARSERCV_OCR_SMALL_REC_MODEL,
        UIPARSERCV_OCR_SMALL_REC_CONFIG
      }
    };

    for (const auto& combo : combos) {
      uiparsercv::pipeline::PipelineOptions options;
      options.icon.model_path = UIPARSERCV_ICON_MODEL;
      options.ocr_detector.model_path = combo.det_model;
      options.ocr_recognizer.model_path = combo.rec_model;
      options.ocr_recognizer.config_path = combo.rec_config;

      uiparsercv::pipeline::PipelineRunner runner(options);
      const auto result = runner.run(image);

      const auto json_path = output_dir / (combo.name + ".json");
      const auto overlay_path = output_dir / (combo.name + ".overlay.png");
      const auto meta_path = output_dir / (combo.name + ".meta.txt");

      write_json_file(json_path, result);
      uiparsercv::exporter::write_debug_overlay(
          image,
          result,
          uiparsercv::exporter::DebugOverlayOptions{overlay_path, meta_path});

      std::cout << combo.name
                << ": icons=" << result.stats.icon_count
                << " text_regions=" << result.stats.text_region_count
                << " candidates=" << result.stats.candidate_count
                << " json=" << json_path.string()
                << " overlay=" << overlay_path.string()
                << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "model_combo_compare failed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
