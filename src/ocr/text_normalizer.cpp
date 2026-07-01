#include "uiparsercv/ocr/text_normalizer.hpp"

#include <cctype>
#include <string>
#include <utility>

namespace uiparsercv::ocr {
namespace {

bool is_space(unsigned char ch) {
  return std::isspace(ch) != 0;
}

std::string trim_and_collapse_ascii_space(const std::string& text) {
  std::string out;
  out.reserve(text.size());

  bool pending_space = false;
  bool wrote_any = false;
  for (const unsigned char ch : text) {
    if (is_space(ch)) {
      if (wrote_any) {
        pending_space = true;
      }
      continue;
    }

    if (pending_space) {
      out.push_back(' ');
      pending_space = false;
    }
    out.push_back(static_cast<char>(ch));
    wrote_any = true;
  }

  return out;
}

} // namespace

NormalizedText normalize_text(std::string raw_text) {
  NormalizedText result;
  result.raw = std::move(raw_text);
  result.normalized = trim_and_collapse_ascii_space(result.raw);
  return result;
}

} // namespace uiparsercv::ocr
