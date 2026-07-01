#include "uiparsercv/export/json_exporter.hpp"

#include <ostream>
#include <string>

namespace uiparsercv::exporter {
namespace {

void indent(std::ostream& out, int spaces) {
  for (int i = 0; i < spaces; ++i) {
    out << ' ';
  }
}

void json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  out << '"';
}

void rect_json(std::ostream& out, const RectF& rect) {
  out << "{\"x\": " << rect.x
      << ", \"y\": " << rect.y
      << ", \"width\": " << rect.width
      << ", \"height\": " << rect.height << '}';
}

void tree_rect_json(std::ostream& out, const tree::Rect& rect) {
  out << "{\"x\": " << rect.x
      << ", \"y\": " << rect.y
      << ", \"width\": " << rect.width
      << ", \"height\": " << rect.height << '}';
}

void candidate_json(std::ostream& out, const pipeline::UiElementCandidate& candidate, int level) {
  indent(out, level);
  out << "{\n";
  indent(out, level + 2);
  out << "\"kind\": ";
  json_string(out, candidate.kind == pipeline::UiElementKind::Icon ? "icon" : "text");
  out << ",\n";

  indent(out, level + 2);
  out << "\"box\": ";
  rect_json(out, candidate.box);
  out << ",\n";

  indent(out, level + 2);
  out << "\"detection_score\": " << candidate.detection_score << ",\n";

  indent(out, level + 2);
  out << "\"text\": ";
  json_string(out, candidate.text);
  out << ",\n";

  indent(out, level + 2);
  out << "\"text_confidence\": " << candidate.text_confidence << ",\n";

  indent(out, level + 2);
  out << "\"interactive\": " << (candidate.interactive ? "true" : "false") << ",\n";

  indent(out, level + 2);
  out << "\"source\": ";
  json_string(out, candidate.source);
  out << '\n';

  indent(out, level);
  out << '}';
}

void tree_json(std::ostream& out, const tree::TreeNode& node, int level) {
  indent(out, level);
  out << "{\n";
  indent(out, level + 2);
  out << "\"id\": " << node.box.id << ",\n";

  indent(out, level + 2);
  out << "\"label\": ";
  json_string(out, node.box.label);
  out << ",\n";

  indent(out, level + 2);
  out << "\"rect\": ";
  tree_rect_json(out, node.box.rect);
  out << ",\n";

  indent(out, level + 2);
  out << "\"children\": [";
  if (!node.children.empty()) {
    out << '\n';
  }

  for (std::size_t i = 0; i < node.children.size(); ++i) {
    tree_json(out, node.children[i], level + 4);
    if (i + 1 < node.children.size()) {
      out << ',';
    }
    out << '\n';
  }

  if (!node.children.empty()) {
    indent(out, level + 2);
  }
  out << "]\n";

  indent(out, level);
  out << '}';
}

} // namespace

void write_json(std::ostream& out, const pipeline::PipelineResult& result) {
  out << "{\n";

  indent(out, 2);
  out << "\"image\": {\"width\": " << result.stats.image_width
      << ", \"height\": " << result.stats.image_height << "},\n";

  indent(out, 2);
  out << "\"stats\": {\"icons\": " << result.stats.icon_count
      << ", \"text_regions\": " << result.stats.text_region_count
      << ", \"candidates\": " << result.stats.candidate_count << "},\n";

  indent(out, 2);
  out << "\"candidates\": [";
  if (!result.candidates.empty()) {
    out << '\n';
  }
  for (std::size_t i = 0; i < result.candidates.size(); ++i) {
    candidate_json(out, result.candidates[i], 4);
    if (i + 1 < result.candidates.size()) {
      out << ',';
    }
    out << '\n';
  }
  if (!result.candidates.empty()) {
    indent(out, 2);
  }
  out << "],\n";

  indent(out, 2);
  out << "\"tree\": \n";
  tree_json(out, result.tree, 2);
  out << '\n';

  out << "}\n";
}

} // namespace uiparsercv::exporter

