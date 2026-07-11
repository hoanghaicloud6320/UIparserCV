#include "uiparsercv/tree/box_tree.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace uiparsercv::tree {
namespace {

void write_indent(std::ostream& out, int indent) {
  for (int i = 0; i < indent; ++i) {
    out << ' ';
  }
}

void write_json_string(std::ostream& out, const std::string& value) {
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

} // namespace

bool contains(const Rect& outer, const Rect& inner) {
  return outer.x <= inner.x && outer.y <= inner.y &&
         outer.x + outer.width >= inner.x + inner.width &&
         outer.y + outer.height >= inner.y + inner.height;
}

int area(const Rect& rect) {
  if (rect.width < 0 || rect.height < 0) {
    throw std::invalid_argument("rectangle dimensions must be non-negative");
  }

  return rect.width * rect.height;
}

TreeNode build_containment_tree(std::vector<Box> boxes) {
  std::sort(boxes.begin(), boxes.end(), [](const Box& lhs, const Box& rhs) {
    return area(lhs.rect) > area(rhs.rect);
  });

  std::vector<int> parents(boxes.size(), -1);
  for (std::size_t child = 0; child < boxes.size(); ++child) {
    int parent = -1;
    int parent_area = 0;

    for (std::size_t candidate = 0; candidate < child; ++candidate) {
      const int candidate_area = area(boxes[candidate].rect);
      const int child_area = area(boxes[child].rect);
      if (candidate_area > child_area &&
          contains(boxes[candidate].rect, boxes[child].rect)) {
        if (parent == -1 || candidate_area < parent_area) {
          parent = static_cast<int>(candidate);
          parent_area = candidate_area;
        }
      }
    }

    parents[child] = parent;
  }

  std::function<TreeNode(std::size_t)> make_node = [&](std::size_t index) {
    TreeNode node{boxes[index], {}};

    for (std::size_t child = 0; child < boxes.size(); ++child) {
      if (parents[child] == static_cast<int>(index)) {
        node.children.push_back(make_node(child));
      }
    }

    return node;
  };

  TreeNode root{Box{0, Rect{0, 0, 0, 0}, "root"}, {}};
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    if (parents[index] == -1) {
      root.children.push_back(make_node(index));
    }
  }

  return root;
}

void write_json(std::ostream& out, const TreeNode& node, int indent) {
  write_indent(out, indent);
  out << "{\n";

  write_indent(out, indent + 2);
  out << "\"id\": " << node.box.id << ",\n";

  write_indent(out, indent + 2);
  out << "\"label\": ";
  write_json_string(out, node.box.label);
  out << ",\n";

  write_indent(out, indent + 2);
  out << "\"rect\": {\"x\": " << node.box.rect.x
      << ", \"y\": " << node.box.rect.y
      << ", \"width\": " << node.box.rect.width
      << ", \"height\": " << node.box.rect.height << "},\n";

  write_indent(out, indent + 2);
  out << "\"children\": [";
  if (!node.children.empty()) {
    out << '\n';
  }

  for (std::size_t i = 0; i < node.children.size(); ++i) {
    write_json(out, node.children[i], indent + 4);
    if (i + 1 < node.children.size()) {
      out << ',';
    }
    out << '\n';
  }

  if (!node.children.empty()) {
    write_indent(out, indent + 2);
  }
  out << "]\n";

  write_indent(out, indent);
  out << '}';
}

} // namespace uiparsercv::tree
