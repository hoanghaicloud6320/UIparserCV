#include "uiparsercv/export/debug_overlay.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace uiparsercv::exporter {
namespace {

constexpr int kScreenNodeId = 1;
constexpr int kFirstCandidateNodeId = 2;

const pipeline::UiElementCandidate* candidate_for_node_id(
    const pipeline::PipelineResult& result,
    int node_id) {
  const int index = node_id - kFirstCandidateNodeId;
  if (index < 0 || static_cast<std::size_t>(index) >= result.candidates.size()) {
    return nullptr;
  }
  return &result.candidates[static_cast<std::size_t>(index)];
}

std::string kind_name(const pipeline::UiElementCandidate& candidate) {
  return candidate.kind == pipeline::UiElementKind::Icon ? "icon" : "text";
}

cv::Scalar color_for(const pipeline::UiElementCandidate& candidate) {
  return candidate.kind == pipeline::UiElementKind::Icon
      ? cv::Scalar(210, 120, 25)
      : cv::Scalar(25, 95, 230);
}

void ensure_parent_dir(const std::filesystem::path& path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

void draw_label(cv::Mat& image, const std::string& text, int x, int y, cv::Scalar color) {
  constexpr double font_scale = 0.43;
  constexpr int thickness = 1;
  int baseline = 0;
  const auto text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

  const int left = std::clamp(x, 0, std::max(0, image.cols - text_size.width - 8));
  const int top = std::clamp(y - text_size.height - 8, 0, std::max(0, image.rows - text_size.height - 8));
  cv::rectangle(
      image,
      {left, top, text_size.width + 8, text_size.height + baseline + 6},
      color,
      cv::FILLED,
      cv::LINE_AA);
  cv::putText(
      image,
      text,
      {left + 4, top + text_size.height + 1},
      cv::FONT_HERSHEY_SIMPLEX,
      font_scale,
      cv::Scalar(255, 255, 255),
      thickness,
      cv::LINE_AA);
}

void draw_tree_node(
    cv::Mat& image,
    const tree::TreeNode& node,
    const pipeline::PipelineResult& result) {
  if (node.box.id != 0 && node.box.id != kScreenNodeId) {
    const auto* candidate = candidate_for_node_id(result, node.box.id);
    const cv::Scalar color = candidate != nullptr ? color_for(*candidate) : cv::Scalar(80, 80, 80);
    const cv::Rect rect{
        node.box.rect.x,
        node.box.rect.y,
        std::max(1, node.box.rect.width),
        std::max(1, node.box.rect.height)};
    cv::rectangle(image, rect, color, 2, cv::LINE_AA);

    std::string label = "#" + std::to_string(node.box.id);
    if (candidate != nullptr) {
      label += " " + kind_name(*candidate);
    }
    draw_label(image, label, rect.x, rect.y, color);
  }

  for (const auto& child : node.children) {
    draw_tree_node(image, child, result);
  }
}

void write_tree_metadata(
    std::ostream& out,
    const tree::TreeNode& node,
    const pipeline::PipelineResult& result,
    int depth) {
  for (int i = 0; i < depth; ++i) {
    out << "  ";
  }

  out << "- #" << node.box.id << ' ' << node.box.label
      << " rect=[" << node.box.rect.x << ',' << node.box.rect.y << ','
      << node.box.rect.width << ',' << node.box.rect.height << ']'
      << " children=" << node.children.size();

  if (const auto* candidate = candidate_for_node_id(result, node.box.id)) {
    out << " kind=" << kind_name(*candidate)
        << " det=" << std::fixed << std::setprecision(3) << candidate->detection_score
        << " text_conf=" << candidate->text_confidence
        << " interactive=" << (candidate->interactive ? "true" : "false")
        << " source=" << candidate->source
        << " text='" << candidate->text << "'"
        << " raw_text='" << candidate->raw_text << "'"
        << " normalized_text='" << candidate->normalized_text << "'";
  }
  out << '\n';

  for (const auto& child : node.children) {
    write_tree_metadata(out, child, result, depth + 1);
  }
}

void draw_legend(cv::Mat& image) {
  cv::rectangle(image, {8, 8, 220, 54}, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
  cv::rectangle(image, {8, 8, 220, 54}, cv::Scalar(220, 224, 230), 1, cv::LINE_AA);
  cv::rectangle(image, {18, 20, 18, 12}, cv::Scalar(25, 95, 230), 2, cv::LINE_AA);
  cv::putText(
      image,
      "text",
      {44, 31},
      cv::FONT_HERSHEY_SIMPLEX,
      0.4,
      cv::Scalar(35, 40, 48),
      1,
      cv::LINE_AA);
  cv::rectangle(image, {18, 41, 18, 12}, cv::Scalar(210, 120, 25), 2, cv::LINE_AA);
  cv::putText(
      image,
      "icon / interactive",
      {44, 52},
      cv::FONT_HERSHEY_SIMPLEX,
      0.4,
      cv::Scalar(35, 40, 48),
      1,
      cv::LINE_AA);
}

} // namespace

void write_debug_overlay(
    const cv::Mat& bgr_image,
    const pipeline::PipelineResult& result,
    const DebugOverlayOptions& options) {
  if (bgr_image.empty()) {
    throw std::runtime_error("debug overlay received an empty image");
  }
  if (options.image_path.empty() && options.metadata_path.empty()) {
    return;
  }

  if (!options.image_path.empty()) {
    ensure_parent_dir(options.image_path);
    cv::Mat overlay = bgr_image.clone();
    draw_tree_node(overlay, result.tree, result);
    draw_legend(overlay);
    if (!cv::imwrite(options.image_path.string(), overlay)) {
      throw std::runtime_error("failed to write debug image: " + options.image_path.string());
    }
  }

  if (!options.metadata_path.empty()) {
    ensure_parent_dir(options.metadata_path);
    std::ofstream out(options.metadata_path);
    if (!out) {
      throw std::runtime_error("failed to write debug metadata: " + options.metadata_path.string());
    }

    out << "image: " << result.stats.image_width << 'x' << result.stats.image_height << '\n';
    out << "stats: icons=" << result.stats.icon_count
        << ", text_regions=" << result.stats.text_region_count
        << ", candidates=" << result.stats.candidate_count << '\n';
    out << '\n';
    out << "TREE + BOX METADATA\n\n";
    write_tree_metadata(out, result.tree, result, 0);
  }
}

} // namespace uiparsercv::exporter
