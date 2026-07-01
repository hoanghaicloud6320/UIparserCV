#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Rect {
  int x{};
  int y{};
  int width{};
  int height{};
};

struct Node {
  int id{};
  int parent{};
  Rect rect{};
  std::string role;
  std::string text;
  bool interactive{};
};

struct CaseSpec {
  std::string name;
  std::string family;
  int width{};
  int height{};
  cv::Scalar background;
  std::vector<Node> nodes;
  cv::Mat image;
};

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
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
  return out.str();
}

std::string rect_json(const Rect& rect) {
  std::ostringstream out;
  out << "{\"x\":" << rect.x << ",\"y\":" << rect.y << ",\"width\":" << rect.width
      << ",\"height\":" << rect.height << "}";
  return out.str();
}

class Painter {
public:
  Painter(int width, int height, cv::Scalar background)
      : image_(height, width, CV_8UC3, background) {}

  cv::Mat& image() { return image_; }

  void panel(Rect r, cv::Scalar fill, cv::Scalar stroke = cv::Scalar(222, 226, 232),
             int radius = 8) {
    rounded_rect(r, fill, true, radius, 1);
    rounded_rect(r, stroke, false, radius, 1);
  }

  void button(Rect r, const std::string& text, cv::Scalar fill, cv::Scalar stroke,
              cv::Scalar text_color = cv::Scalar(48, 54, 63)) {
    rounded_rect(r, fill, true, 10, 1);
    rounded_rect(r, stroke, false, 10, 1);
    label({r.x + 14, r.y + r.height / 2 + 5}, text, 0.45, text_color, 1);
  }

  void input(Rect r, const std::string& text) {
    panel(r, cv::Scalar(255, 255, 255), cv::Scalar(215, 220, 228), 12);
    label({r.x + 16, r.y + r.height / 2 + 5}, text, 0.47, cv::Scalar(135, 139, 146), 1);
  }

  void label(cv::Point origin, const std::string& text, double scale,
             cv::Scalar color = cv::Scalar(36, 41, 48), int thickness = 1) {
    cv::putText(image_, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv::LINE_AA);
  }

  void icon_circle(cv::Point center, int radius, cv::Scalar fill, cv::Scalar stroke,
                   const std::string& glyph) {
    cv::circle(image_, center, radius, fill, cv::FILLED, cv::LINE_AA);
    cv::circle(image_, center, radius, stroke, 1, cv::LINE_AA);
    if (!glyph.empty()) {
      const double scale = 0.45;
      int baseline = 0;
      const auto size = cv::getTextSize(glyph, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
      cv::putText(image_, glyph,
                  {center.x - size.width / 2, center.y + size.height / 2},
                  cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(43, 49, 57), 1, cv::LINE_AA);
    }
  }

  void checkbox(Rect r, bool checked) {
    cv::rectangle(image_, {r.x, r.y, r.width, r.height}, cv::Scalar(116, 135, 160), 1,
                  cv::LINE_AA);
    if (checked) {
      cv::line(image_, {r.x + 4, r.y + r.height / 2}, {r.x + r.width / 2, r.y + r.height - 5},
               cv::Scalar(20, 132, 92), 2, cv::LINE_AA);
      cv::line(image_, {r.x + r.width / 2, r.y + r.height - 5}, {r.x + r.width - 4, r.y + 4},
               cv::Scalar(20, 132, 92), 2, cv::LINE_AA);
    }
  }

private:
  void rounded_rect(Rect r, cv::Scalar color, bool fill, int /*radius*/, int thickness) {
    const int line = fill ? cv::FILLED : thickness;
    cv::rectangle(image_, {r.x, r.y, r.width, r.height}, color, line, cv::LINE_AA);
  }

  cv::Mat image_;
};

Node node(int id, int parent, Rect rect, std::string role, std::string text = {},
          bool interactive = false) {
  return {id, parent, rect, std::move(role), std::move(text), interactive};
}

void add_root(CaseSpec& spec) {
  spec.nodes.push_back(node(0, -1, {0, 0, spec.width, spec.height}, "screen", spec.name, false));
}

CaseSpec make_case(int index) {
  static const std::vector<std::string> families = {
      "chat",     "settings", "dashboard", "login",   "mail",    "calendar",
      "shop",     "music",    "files",     "editor",  "map",     "profile"};
  const std::string family = families[static_cast<size_t>(index % families.size())];
  const bool mobile = (index % 3) == 0;
  const int width = mobile ? 390 : 933;
  const int height = mobile ? 780 : 401;
  CaseSpec spec{"synthetic_" + std::to_string(index + 1) + "_" + family,
                family,
                width,
                height,
                cv::Scalar(247, 248, 250),
                {},
                {}};
  Painter p(width, height, spec.background);
  add_root(spec);
  int id = 1;

  if (family == "chat") {
    p.label({mobile ? 76 : 330, 64}, "How can I help you?", 0.78, cv::Scalar(25, 29, 35), 2);
    spec.nodes.push_back(node(id++, 0, {mobile ? 76 : 330, 42, 310, 34}, "title",
                              "How can I help you?"));
    const Rect bar{mobile ? 24 : 92, mobile ? 130 : 160, mobile ? 342 : 770, 54};
    p.input(bar, "Ask anything");
    const int searchbox_id = id++;
    spec.nodes.push_back(node(searchbox_id, 0, bar, "searchbox", "Ask anything", true));
    p.icon_circle({bar.x + 27, bar.y + 27}, 16, cv::Scalar(255, 255, 255),
                  cv::Scalar(190, 196, 205), "+");
    spec.nodes.push_back(node(id++, searchbox_id, {bar.x + 11, bar.y + 11, 32, 32}, "icon", "+",
                              true));
    p.button({mobile ? 42 : 240, mobile ? 230 : 238, 112, 42}, "Create", cv::Scalar(255, 255, 255),
             cv::Scalar(220, 224, 230));
    spec.nodes.push_back(node(id++, 0, {mobile ? 42 : 240, mobile ? 230 : 238, 112, 42}, "button",
                              "Create", true));
    p.button({mobile ? 164 : 362, mobile ? 230 : 238, 172, 42}, "Edit image",
             cv::Scalar(255, 255, 255), cv::Scalar(220, 224, 230));
    spec.nodes.push_back(node(id++, 0, {mobile ? 164 : 362, mobile ? 230 : 238, 172, 42},
                              "button", "Edit image", true));
  } else if (family == "settings") {
    p.panel({24, 24, width - 48, height - 48}, cv::Scalar(255, 255, 255));
    spec.nodes.push_back(node(id++, 0, {24, 24, width - 48, height - 48}, "panel", "Settings"));
    p.label({48, 70}, "Settings", 0.8, cv::Scalar(35, 39, 47), 2);
    spec.nodes.push_back(node(id++, 1, {48, 42, 150, 38}, "title", "Settings"));
    for (int row = 0; row < 5; ++row) {
      const int y = 104 + row * 54;
      p.label({56, y + 25}, "Option " + std::to_string(row + 1), 0.48);
      p.checkbox({width - 92, y + 7, 22, 22}, row % 2 == 0);
      spec.nodes.push_back(node(id++, 1, {48, y, width - 96, 38}, "setting_row",
                                "Option " + std::to_string(row + 1), true));
    }
  } else if (family == "dashboard") {
    p.label({32, 52}, "Revenue", 0.72, cv::Scalar(30, 35, 42), 2);
    spec.nodes.push_back(node(id++, 0, {32, 25, 150, 36}, "title", "Revenue"));
    for (int col = 0; col < 3; ++col) {
      const Rect card{32 + col * ((width - 96) / 3), 82, (width - 128) / 3, 110};
      p.panel(card, cv::Scalar(255, 255, 255));
      p.label({card.x + 18, card.y + 34}, col == 0 ? "Sales" : col == 1 ? "Users" : "Orders",
              0.5, cv::Scalar(90, 97, 108));
      p.label({card.x + 18, card.y + 78}, col == 0 ? "$12.4k" : col == 1 ? "8,210" : "342",
              0.72, cv::Scalar(32, 37, 44), 2);
      spec.nodes.push_back(node(id++, 0, card, "metric_card",
                                col == 0 ? "Sales" : col == 1 ? "Users" : "Orders"));
    }
    p.panel({32, 220, width - 64, 128}, cv::Scalar(255, 255, 255));
    spec.nodes.push_back(node(id++, 0, {32, 220, width - 64, 128}, "chart", "Weekly chart"));
  } else if (family == "login") {
    const Rect card{mobile ? 34 : width / 2 - 180, 70, 360, 250};
    p.panel(card, cv::Scalar(255, 255, 255));
    spec.nodes.push_back(node(id++, 0, card, "dialog", "Sign in"));
    p.label({card.x + 28, card.y + 44}, "Sign in", 0.75, cv::Scalar(32, 37, 44), 2);
    spec.nodes.push_back(node(id++, 1, {card.x + 28, card.y + 18, 120, 36}, "title", "Sign in"));
    p.input({card.x + 28, card.y + 78, card.width - 56, 42}, "Email");
    spec.nodes.push_back(node(id++, 1, {card.x + 28, card.y + 78, card.width - 56, 42}, "textbox",
                              "Email", true));
    p.input({card.x + 28, card.y + 134, card.width - 56, 42}, "Password");
    spec.nodes.push_back(node(id++, 1, {card.x + 28, card.y + 134, card.width - 56, 42}, "textbox",
                              "Password", true));
    p.button({card.x + 28, card.y + 194, card.width - 56, 42}, "Continue",
             cv::Scalar(43, 105, 218), cv::Scalar(43, 105, 218), cv::Scalar(255, 255, 255));
    spec.nodes.push_back(node(id++, 1, {card.x + 28, card.y + 194, card.width - 56, 42}, "button",
                              "Continue", true));
  } else {
    p.panel({22, 22, width - 44, 58}, cv::Scalar(255, 255, 255));
    p.label({44, 58}, family + " app", 0.67, cv::Scalar(32, 37, 44), 2);
    spec.nodes.push_back(node(id++, 0, {22, 22, width - 44, 58}, "toolbar", family + " app"));
    p.button({width - 132, 32, 88, 38}, "New", cv::Scalar(43, 105, 218),
             cv::Scalar(43, 105, 218), cv::Scalar(255, 255, 255));
    spec.nodes.push_back(node(id++, 1, {width - 132, 32, 88, 38}, "button", "New", true));
    const int cols = mobile ? 1 : 3;
    const int gap = 18;
    const int card_w = (width - 44 - gap * (cols + 1)) / cols;
    for (int i = 0; i < (mobile ? 5 : 6); ++i) {
      const int col = i % cols;
      const int row = i / cols;
      const Rect card{22 + gap + col * (card_w + gap), 104 + row * 92, card_w, 72};
      p.panel(card, cv::Scalar(255, 255, 255));
      p.icon_circle({card.x + 26, card.y + 36}, 15, cv::Scalar(239, 243, 248),
                    cv::Scalar(205, 213, 224), std::to_string(i + 1));
      p.label({card.x + 54, card.y + 32}, family + " item " + std::to_string(i + 1), 0.43);
      p.label({card.x + 54, card.y + 54}, "Secondary text", 0.35, cv::Scalar(120, 128, 138));
      spec.nodes.push_back(node(id++, 0, card, "list_item",
                                family + " item " + std::to_string(i + 1), true));
    }
  }

  spec.image = p.image().clone();
  return spec;
}

void write_case_json(const std::filesystem::path& path, const CaseSpec& spec,
                     const std::string& image_name) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
  out << "{\n";
  out << "  \"name\": \"" << json_escape(spec.name) << "\",\n";
  out << "  \"family\": \"" << json_escape(spec.family) << "\",\n";
  out << "  \"image\": \"" << json_escape(image_name) << "\",\n";
  out << "  \"size\": {\"width\": " << spec.width << ", \"height\": " << spec.height << "},\n";
  out << "  \"nodes\": [\n";
  for (size_t i = 0; i < spec.nodes.size(); ++i) {
    const auto& n = spec.nodes[i];
    out << "    {\"id\": " << n.id << ", \"parent\": " << n.parent
        << ", \"role\": \"" << json_escape(n.role) << "\", \"text\": \""
        << json_escape(n.text) << "\", \"interactive\": " << (n.interactive ? "true" : "false")
        << ", \"rect\": " << rect_json(n.rect) << "}";
    out << (i + 1 == spec.nodes.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

void write_manifest(const std::filesystem::path& path, const std::vector<CaseSpec>& specs) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
  out << "{\n  \"version\": 1,\n  \"count\": " << specs.size() << ",\n  \"cases\": [\n";
  for (size_t i = 0; i < specs.size(); ++i) {
    out << "    {\"name\": \"" << json_escape(specs[i].name) << "\", \"family\": \""
        << json_escape(specs[i].family) << "\", \"image\": \"" << specs[i].name
        << ".png\", \"ground_truth\": \"" << specs[i].name << ".json\"}";
    out << (i + 1 == specs.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path output_dir =
        argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("synthetic_ui_corpus");
    const int count = argc > 2 ? std::stoi(argv[2]) : 36;
    if (count <= 0) {
      throw std::runtime_error("case count must be positive");
    }

    std::filesystem::create_directories(output_dir);
    std::vector<CaseSpec> specs;
    specs.reserve(static_cast<size_t>(count));

    for (int i = 0; i < count; ++i) {
      auto spec = make_case(i);
      const auto image_name = spec.name + ".png";
      const auto json_name = spec.name + ".json";
      if (!cv::imwrite((output_dir / image_name).string(), spec.image)) {
        throw std::runtime_error("cannot write image " + (output_dir / image_name).string());
      }
      write_case_json(output_dir / json_name, spec, image_name);
      specs.push_back(std::move(spec));
    }

    write_manifest(output_dir / "manifest.json", specs);
    std::cout << "generated " << specs.size() << " synthetic UI cases in "
              << output_dir.string() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "synthetic_ui_corpus failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
