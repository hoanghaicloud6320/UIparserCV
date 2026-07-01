#include "uiparsercv/export/json_exporter.hpp"
#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
  std::filesystem::path input;
  std::filesystem::path output{"ui_tree.json"};
};

void print_usage(std::ostream& out) {
  out << "usage: uiparsercv_pipeline <input-image> [--out output.json]\n";
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
    options.icon.model_path = UIPARSERCV_ICON_MODEL;
    options.ocr_detector.model_path = UIPARSERCV_OCR_DET_MODEL;
    options.ocr_recognizer.model_path = UIPARSERCV_OCR_REC_MODEL;
    options.ocr_recognizer.config_path = UIPARSERCV_OCR_REC_CONFIG;

    uiparsercv::pipeline::PipelineRunner runner(options);
    const auto result = runner.run_file(cli.input);

    std::ofstream output(cli.output);
    if (!output) {
      throw std::runtime_error("failed to open output file: " + cli.output.string());
    }
    uiparsercv::exporter::write_json(output, result);

    std::cout << "image: " << result.stats.image_width << "x" << result.stats.image_height << '\n'
              << "icons: " << result.stats.icon_count << '\n'
              << "text_regions: " << result.stats.text_region_count << '\n'
              << "candidates: " << result.stats.candidate_count << '\n'
              << "output: " << cli.output.string() << '\n';

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pipeline error: " << error.what() << '\n';
    print_usage(std::cerr);
    return 1;
  }
}

