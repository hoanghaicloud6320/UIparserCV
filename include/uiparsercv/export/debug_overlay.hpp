#pragma once

#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <opencv2/core.hpp>

#include <filesystem>

namespace uiparsercv::exporter {

struct DebugOverlayOptions {
  std::filesystem::path image_path;
  std::filesystem::path metadata_path;
};

void write_debug_overlay(
    const cv::Mat& bgr_image,
    const pipeline::PipelineResult& result,
    const DebugOverlayOptions& options);

} // namespace uiparsercv::exporter
