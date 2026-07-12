#include "uiparsercv/export/debug_overlay.hpp"

#include <opencv2/freetype.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <opencv2/core/cvstd.hpp>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>

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
  if (candidate.kind == pipeline::UiElementKind::ModelProposal) {
    return "model";
  }
  if (candidate.kind == pipeline::UiElementKind::Icon) {
    return "icon";
  }
  if (candidate.kind == pipeline::UiElementKind::Text) {
    return "text";
  }
  return candidate.kind == pipeline::UiElementKind::VisualContainer ? "container" : "group";
}

cv::Scalar color_for(const pipeline::UiElementCandidate& candidate) {
  if (candidate.kind == pipeline::UiElementKind::ModelProposal) {
    return cv::Scalar(40, 200, 40);
  }
  if (candidate.kind == pipeline::UiElementKind::Icon) {
    return cv::Scalar(210, 120, 25);
  }
  if (candidate.kind == pipeline::UiElementKind::Text) {
    return cv::Scalar(25, 95, 230);
  }
  return candidate.kind == pipeline::UiElementKind::VisualContainer
      ? cv::Scalar(40, 180, 210)
      : cv::Scalar(190, 70, 190);
}

void draw_associations(
    cv::Mat& image,
    const pipeline::PipelineResult& result) {
  for (const auto& association : result.associations) {
    if (association.proposal_index >= result.candidates.size() ||
        association.text_index >= result.candidates.size()) continue;
    const auto& proposal = result.candidates[association.proposal_index].box;
    const auto& text = result.candidates[association.text_index].box;
    const cv::Point from{
        static_cast<int>(std::round(proposal.x + proposal.width / 2.0F)),
        static_cast<int>(std::round(proposal.y + proposal.height / 2.0F))};
    const cv::Point to{
        static_cast<int>(std::round(text.x + text.width / 2.0F)),
        static_cast<int>(std::round(text.y + text.height / 2.0F))};
    cv::line(image, from, to, cv::Scalar(230, 80, 220), 1, cv::LINE_AA);
  }
}

bool is_group_node(const tree::TreeNode& node) {
  return node.box.label.rfind("group:", 0) == 0;
}

cv::Scalar group_color(const tree::TreeNode& node) {
  if (node.box.label == "group: input_bar") {
    return cv::Scalar(45, 190, 80);
  }
  if (node.box.label == "group: button") {
    return cv::Scalar(210, 55, 190);
  }
  return cv::Scalar(40, 175, 220);
}

cv::Scalar relation_color(const tree::TreeNode& parent) {
  return is_group_node(parent) ? group_color(parent) : cv::Scalar(20, 190, 235);
}

std::string compact_label_text(const std::string& text, std::size_t max_chars = 18) {
  std::string out;
  out.reserve(text.size());

  bool pending_space = false;
  std::size_t chars = 0;
  for (std::size_t i = 0; i < text.size();) {
    const auto ch = static_cast<unsigned char>(text[i]);
    if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ') {
      pending_space = !out.empty();
      ++i;
      continue;
    }
    if (pending_space) {
      out.push_back(' ');
      pending_space = false;
    }

    std::size_t char_len = 1;
    if ((ch & 0xE0U) == 0xC0U) {
      char_len = 2;
    } else if ((ch & 0xF0U) == 0xE0U) {
      char_len = 3;
    } else if ((ch & 0xF8U) == 0xF0U) {
      char_len = 4;
    }
    if (i + char_len > text.size()) {
      break;
    }
    out.append(text, i, char_len);
    i += char_len;
    ++chars;

    if (chars >= max_chars && i < text.size()) {
      out += "...";
      break;
    }
  }

  return out;
}

std::vector<std::uint32_t> utf8_codepoints(const std::string& text) {
  std::vector<std::uint32_t> result;
  for (std::size_t i = 0; i < text.size();) {
    const auto ch = static_cast<unsigned char>(text[i]);
    if (ch < 0x80) {
      result.push_back(ch);
      ++i;
    } else if ((ch & 0xE0U) == 0xC0U && i + 1 < text.size()) {
      result.push_back(((ch & 0x1FU) << 6U) | (static_cast<unsigned char>(text[i + 1]) & 0x3FU));
      i += 2;
    } else if ((ch & 0xF0U) == 0xE0U && i + 2 < text.size()) {
      result.push_back(
          ((ch & 0x0FU) << 12U) |
          ((static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 6U) |
          (static_cast<unsigned char>(text[i + 2]) & 0x3FU));
      i += 3;
    } else if ((ch & 0xF8U) == 0xF0U && i + 3 < text.size()) {
      result.push_back(
          ((ch & 0x07U) << 18U) |
          ((static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 12U) |
          ((static_cast<unsigned char>(text[i + 2]) & 0x3FU) << 6U) |
          (static_cast<unsigned char>(text[i + 3]) & 0x3FU));
      i += 4;
    } else {
      ++i;
    }
  }
  return result;
}

void ensure_parent_dir(const std::filesystem::path& path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

class TextDrawer {
public:
  TextDrawer() {
    latin_ = load_font("C:/Windows/Fonts/arial.ttf");
    japanese_ = load_font("C:/Windows/Fonts/YuGothR.ttc");
    chinese_ = load_font("C:/Windows/Fonts/msyh.ttc");
    korean_ = load_font("C:/Windows/Fonts/malgun.ttf");
  }

  cv::Size measure(const std::string& text, int* baseline) const {
    if (const auto font = font_for(text)) {
      return font->getTextSize(text, kFontHeight, kThickness, baseline);
    }
    return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, kHersheyScale, kThickness, baseline);
  }

  void draw(cv::Mat& image, const std::string& text, cv::Point origin, cv::Scalar color) const {
    if (const auto font = font_for(text)) {
      font->putText(image, text, origin, kFontHeight, color, kThickness, cv::LINE_AA, true);
      return;
    }
    cv::putText(
        image,
        text,
        origin,
        cv::FONT_HERSHEY_SIMPLEX,
        kHersheyScale,
        color,
        kThickness,
        cv::LINE_AA);
  }

private:
  static constexpr int kFontHeight = 14;
  static constexpr int kThickness = 1;
  static constexpr double kHersheyScale = 0.43;

  static cv::Ptr<cv::freetype::FreeType2> load_font(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
      return {};
    }
    try {
      auto font = cv::freetype::createFreeType2();
      font->loadFontData(path.string(), 0);
      return font;
    } catch (const cv::Exception&) {
      return {};
    }
  }

  cv::Ptr<cv::freetype::FreeType2> font_for(const std::string& text) const {
    bool has_cjk = false;
    for (const auto cp : utf8_codepoints(text)) {
      if ((cp >= 0xAC00U && cp <= 0xD7AFU) || (cp >= 0x1100U && cp <= 0x11FFU)) {
        return korean_ ? korean_ : latin_;
      }
      if ((cp >= 0x3040U && cp <= 0x30FFU) || (cp >= 0x31F0U && cp <= 0x31FFU)) {
        return japanese_ ? japanese_ : latin_;
      }
      if (cp >= 0x4E00U && cp <= 0x9FFFU) {
        has_cjk = true;
      }
    }
    if (has_cjk) {
      return chinese_ ? chinese_ : latin_;
    }
    return latin_;
  }

  cv::Ptr<cv::freetype::FreeType2> latin_;
  cv::Ptr<cv::freetype::FreeType2> japanese_;
  cv::Ptr<cv::freetype::FreeType2> chinese_;
  cv::Ptr<cv::freetype::FreeType2> korean_;
};

void draw_label(
    cv::Mat& image,
    const std::string& text,
    int x,
    int y,
    cv::Scalar color,
    const TextDrawer& text_drawer) {
  int baseline = 0;
  const auto text_size = text_drawer.measure(text, &baseline);

  const int left = std::clamp(x, 0, std::max(0, image.cols - text_size.width - 8));
  const int top = std::clamp(y - text_size.height - 8, 0, std::max(0, image.rows - text_size.height - 8));
  cv::rectangle(
      image,
      {left, top, text_size.width + 8, text_size.height + baseline + 6},
      color,
      cv::FILLED,
      cv::LINE_AA);
  text_drawer.draw(image, text, {left + 4, top + text_size.height + 1}, cv::Scalar(255, 255, 255));
}

void draw_tree_node(
    cv::Mat& image,
    const tree::TreeNode& node,
    const pipeline::PipelineResult& result,
    const TextDrawer& text_drawer,
    int parent_id = 0,
    const tree::TreeNode* parent = nullptr) {
  if (node.box.id != 0 && node.box.id != kScreenNodeId) {
    const auto* candidate = candidate_for_node_id(result, node.box.id);
    const bool group = is_group_node(node);
    const cv::Scalar color = candidate != nullptr
        ? color_for(*candidate)
        : (group ? group_color(node) : cv::Scalar(80, 80, 80));
    const cv::Rect rect{
        node.box.rect.x,
        node.box.rect.y,
        std::max(1, node.box.rect.width),
        std::max(1, node.box.rect.height)};
    cv::rectangle(image, rect, color, group ? 4 : 2, cv::LINE_AA);

    if (parent != nullptr && parent->box.id > kScreenNodeId && parent->children.size() <= 8) {
      const auto& parent_rect = parent->box.rect;
      const cv::Point from{
          parent_rect.x + parent_rect.width / 2,
          parent_rect.y + std::min(18, parent_rect.height / 2)};
      const cv::Point to{
          rect.x + rect.width / 2,
          rect.y + rect.height / 2};
      cv::arrowedLine(image, from, to, relation_color(*parent), 2, cv::LINE_AA, 0, 0.025);
    }

    std::string label = "#" + std::to_string(node.box.id);
    if (candidate != nullptr) {
      label += " " + kind_name(*candidate);
      if (parent_id > kScreenNodeId) {
        label += " p#" + std::to_string(parent_id);
      }
      const std::string text = compact_label_text(candidate->text);
      if (!text.empty()) {
        label += ": " + text;
      }
    } else if (group) {
      label += " " + node.box.label;
      label += " [" + std::to_string(node.children.size()) + " children]";
    }
    draw_label(image, label, rect.x, rect.y, color, text_drawer);
  }

  for (const auto& child : node.children) {
    draw_tree_node(image, child, result, text_drawer, node.box.id, &node);
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
  cv::rectangle(image, {8, 8, 300, 138}, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
  cv::rectangle(image, {8, 8, 300, 138}, cv::Scalar(220, 224, 230), 1, cv::LINE_AA);
  cv::rectangle(image, {18, 20, 18, 12}, cv::Scalar(25, 95, 230), 2, cv::LINE_AA);
  cv::putText(
      image,
      "OCR text (geometry preserved)",
      {44, 31},
      cv::FONT_HERSHEY_SIMPLEX,
      0.4,
      cv::Scalar(35, 40, 48),
      1,
      cv::LINE_AA);
  cv::rectangle(image, {18, 41, 18, 12}, cv::Scalar(210, 120, 25), 2, cv::LINE_AA);
  cv::putText(
      image,
      "OmniParser support icon",
      {44, 52},
      cv::FONT_HERSHEY_SIMPLEX,
      0.4,
      cv::Scalar(35, 40, 48),
      1,
      cv::LINE_AA);
  cv::rectangle(image, {18, 62, 18, 12}, cv::Scalar(40, 200, 40), 2, cv::LINE_AA);
  cv::putText(image, "UITag model proposal", {44, 73}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
              cv::Scalar(35, 40, 48), 1, cv::LINE_AA);
  cv::line(image, {18, 87}, {36, 87}, cv::Scalar(230, 80, 220), 2, cv::LINE_AA);
  cv::putText(image, "model - OCR association", {44, 93}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
              cv::Scalar(35, 40, 48), 1, cv::LINE_AA);
  cv::arrowedLine(image, {18, 110}, {36, 110}, cv::Scalar(20, 190, 235), 2, cv::LINE_AA, 0, 0.2);
  cv::putText(
      image,
      "containment parent -> direct child",
      {44, 116},
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
    const TextDrawer text_drawer;
    draw_associations(overlay, result);
    draw_tree_node(overlay, result.tree, result, text_drawer);
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
    out << "stats: uitag_raw=" << result.stats.uitag_raw_count
        << ", uitag=" << result.stats.uitag_count
        << ", support_icons_raw=" << result.stats.icon_count
        << ", text_regions=" << result.stats.text_region_count
        << ", associations=" << result.stats.association_count
        << ", visual_containers=" << result.stats.visual_container_count
        << ", line_rects=" << result.stats.line_rect_count
        << ", inferred_groups=" << result.stats.inferred_group_count
        << ", candidates=" << result.stats.candidate_count << '\n';
    out << '\n';
    out << "RAW UITAG DETECTIONS (BEFORE CROSS-TILE NMS)\n\n";
    for (std::size_t i = 0; i < result.uitag_raw_detections.size(); ++i) {
      const auto& detection = result.uitag_raw_detections[i];
      out << "- uitag_raw[" << i << "] rect=["
          << detection.box.x << ',' << detection.box.y << ','
          << detection.box.width << ',' << detection.box.height << "]"
          << " score=" << std::fixed << std::setprecision(3) << detection.score
          << " class='" << detect::UitagDetector::class_name(detection.class_id) << "'\n";
    }
    out << '\n';
    out << "RAW OMNIPARSER SUPPORT ICON DETECTIONS\n\n";
    for (std::size_t i = 0; i < result.icons.size(); ++i) {
      const auto& detection = result.icons[i];
      out << "- icon_raw[" << i << "] rect=["
          << detection.box.x << ',' << detection.box.y << ','
          << detection.box.width << ',' << detection.box.height << "]"
          << " score=" << std::fixed << std::setprecision(3) << detection.score
          << " label='" << detection.label << "'" << '\n';
    }
    out << '\n';
    out << "MODEL/OCR ASSOCIATIONS\n\n";
    for (const auto& association : result.associations) {
      out << "- candidate[" << association.proposal_index << "] -> candidate["
          << association.text_index << "] relation=" << association.relation
          << " confidence=" << association.confidence << '\n';
    }
    out << '\n';
    out << "RAW OCR REGIONS + RECOGNITION\n\n";
    for (std::size_t i = 0; i < result.text_regions.size(); ++i) {
      const auto& region = result.text_regions[i];
      out << "- ocr_raw[" << i << "] rect=["
          << region.box.x << ',' << region.box.y << ','
          << region.box.width << ',' << region.box.height << "]"
          << " det=" << std::fixed << std::setprecision(3) << region.score;
      if (i < result.recognized_text.size()) {
        out << " rec=" << result.recognized_text[i].confidence
            << " text='" << result.recognized_text[i].text << "'";
      }
      out << '\n';
    }
    out << '\n';
    out << "TREE + BOX METADATA\n\n";
    write_tree_metadata(out, result.tree, result, 0);
  }
}

} // namespace uiparsercv::exporter
