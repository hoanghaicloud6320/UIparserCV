#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace uiparsercv::tree {

struct Rect {
  int x{};
  int y{};
  int width{};
  int height{};
};

struct Box {
  int id{};
  Rect rect{};
  std::string label;
};

struct TreeNode {
  Box box{};
  std::vector<TreeNode> children;
};

bool contains(const Rect& outer, const Rect& inner);
int area(const Rect& rect);
TreeNode build_containment_tree(std::vector<Box> boxes);
void write_json(std::ostream& out, const TreeNode& node, int indent = 0);

} // namespace uiparsercv::tree

