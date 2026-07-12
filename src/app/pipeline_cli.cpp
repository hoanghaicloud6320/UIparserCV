#include "uiparsercv/export/debug_overlay.hpp"
#include "uiparsercv/export/json_exporter.hpp"
#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
  std::filesystem::path input;
  std::filesystem::path output{"ui_tree.json"};
  std::filesystem::path debug_image;
  std::filesystem::path debug_metadata;
  std::filesystem::path icon_model{UIPARSERCV_ICON_MODEL};
  std::filesystem::path uitag_model{UIPARSERCV_UITAG_MODEL};
  std::filesystem::path ocr_det_model{UIPARSERCV_OCR_DET_MODEL};
  std::filesystem::path ocr_rec_model{UIPARSERCV_OCR_REC_MODEL};
  std::filesystem::path ocr_rec_config{UIPARSERCV_OCR_REC_CONFIG};
  bool legacy_icon_support{true};
  bool legacy_heuristics{false};
};

void print_usage(std::ostream& out) {
  out << "usage: uiparsercv_pipeline <input-image> [--out output.json]\n"
      << "       [--debug-image overlay.png] [--debug-meta metadata.txt]\n"
      << "       [--icon-model model.onnx]\n"
      << "       [--uitag-model model.onnx] [--no-legacy-icon-support]\n"
      << "       [--legacy-heuristics]\n"
      << "       [--ocr-det-model det.onnx] [--ocr-rec-model rec.onnx]\n"
      << "       [--ocr-rec-config rec.yml]\n";
}

CliOptions parse_args(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error("missing input image");
  }

  CliOptions options;
  options.input = argv[1];

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--out requires a path");
      }
      options.output = argv[++i];
    } else if (arg == "--debug-image") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--debug-image requires a path");
      }
      options.debug_image = argv[++i];
    } else if (arg == "--debug-meta") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--debug-meta requires a path");
      }
      options.debug_metadata = argv[++i];
    } else if (arg == "--icon-model") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--icon-model requires a path");
      }
      options.icon_model = argv[++i];
    } else if (arg == "--uitag-model") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--uitag-model requires a path");
      }
      options.uitag_model = argv[++i];
    } else if (arg == "--no-legacy-icon-support") {
      options.legacy_icon_support = false;
    } else if (arg == "--legacy-heuristics") {
      options.legacy_heuristics = true;
    } else if (arg == "--ocr-det-model") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--ocr-det-model requires a path");
      }
      options.ocr_det_model = argv[++i];
    } else if (arg == "--ocr-rec-model") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--ocr-rec-model requires a path");
      }
      options.ocr_rec_model = argv[++i];
    } else if (arg == "--ocr-rec-config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--ocr-rec-config requires a path");
      }
      options.ocr_rec_config = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(std::cout);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  return options;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);

    uiparsercv::pipeline::PipelineOptions options;
    options.uitag_model_path = cli.uitag_model;
    options.enable_legacy_icon_support = cli.legacy_icon_support;
    options.enable_legacy_heuristics = cli.legacy_heuristics;
    options.icon.model_path = cli.icon_model;
    options.ocr_detector.model_path = cli.ocr_det_model;
    options.ocr_recognizer.model_path = cli.ocr_rec_model;
    options.ocr_recognizer.config_path = cli.ocr_rec_config;

    uiparsercv::pipeline::PipelineRunner runner(options);
    const cv::Mat image = cv::imread(cli.input.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to load image: " + cli.input.string());
    }
    const auto result = runner.run(image);

    if (!cli.output.parent_path().empty()) {
      std::filesystem::create_directories(cli.output.parent_path());
    }

    std::ofstream output(cli.output);
    if (!output) {
      throw std::runtime_error("failed to open output file: " + cli.output.string());
    }
    uiparsercv::exporter::write_json(output, result);

    uiparsercv::exporter::write_debug_overlay(
        image,
        result,
        uiparsercv::exporter::DebugOverlayOptions{
            cli.debug_image,
            cli.debug_metadata});

    std::cout << "image: " << result.stats.image_width << "x" << result.stats.image_height << '\n'
              << "uitag_raw: " << result.stats.uitag_raw_count << '\n'
              << "uitag: " << result.stats.uitag_count << '\n'
              << "support_icons_raw: " << result.stats.icon_count << '\n'
              << "text_regions: " << result.stats.text_region_count << '\n'
              << "associations: " << result.stats.association_count << '\n'
              << "visual_containers: " << result.stats.visual_container_count << '\n'
              << "line_rects: " << result.stats.line_rect_count << '\n'
              << "inferred_groups: " << result.stats.inferred_group_count << '\n'
              << "candidates: " << result.stats.candidate_count << '\n'
              << "output: " << cli.output.string() << '\n';
    if (!cli.debug_image.empty()) {
      std::cout << "debug_image: " << cli.debug_image.string() << '\n';
    }
    if (!cli.debug_metadata.empty()) {
      std::cout << "debug_meta: " << cli.debug_metadata.string() << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pipeline error: " << error.what() << '\n';
    print_usage(std::cerr);
    return 1;
  }
}
