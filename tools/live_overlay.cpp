#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <windows.h>
#include <windowsx.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr wchar_t kMainClass[] = L"UIparserCVLiveOverlayMain";
constexpr wchar_t kSnipClass[] = L"UIparserCVLiveOverlaySnip";
constexpr wchar_t kTitle[] = L"UIparserCV live-overlay";
constexpr int kNewSnipId = 1001;
constexpr int kExitId = 1002;
constexpr UINT_PTR kSnipTimer = 2001;
constexpr UINT kInferenceDone = WM_APP + 1;
constexpr int kPanelWidth = 400;

struct ScreenRect {
  int x{};
  int y{};
  int width{};
  int height{};
};

struct InferencePayload {
  cv::Mat overlay;
  uiparsercv::pipeline::PipelineStats stats;
  ScreenRect region;
  long long elapsed_ms{};
  std::string error;
  std::string tree_json;
};

struct AppState {
  HWND main_window{};
  HWND new_snip_button{};
  HWND exit_button{};
  HWND tree_json_field{};
  std::shared_ptr<uiparsercv::pipeline::PipelineRunner> runner;
  cv::Mat overlay;
  std::optional<uiparsercv::pipeline::PipelineStats> stats;
  ScreenRect region;
  long long elapsed_ms{};
  bool processing{};
  std::string error;
};

std::wstring utf8_to_wide(const std::string& value) {
  if (value.empty()) return {};
  const int count = MultiByteToWideChar(
      CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
  if (count <= 0) return std::wstring(value.begin(), value.end());
  std::wstring result(static_cast<std::size_t>(count), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), count);
  return result;
}

struct SnipState {
  HWND main_window{};
  ScreenRect screen;
  cv::Mat snapshot;
  POINT anchor{};
  POINT cursor{};
  bool dragging{};
};

AppState* app_state(HWND window) {
  return reinterpret_cast<AppState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
}

cv::Mat capture_screen(const ScreenRect& rect) {
  HDC screen = GetDC(nullptr);
  HDC memory = CreateCompatibleDC(screen);
  BITMAPINFO info{};
  info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  info.bmiHeader.biWidth = rect.width;
  info.bmiHeader.biHeight = -rect.height;
  info.bmiHeader.biPlanes = 1;
  info.bmiHeader.biBitCount = 32;
  info.bmiHeader.biCompression = BI_RGB;
  void* pixels = nullptr;
  HBITMAP bitmap = CreateDIBSection(screen, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
  if (!screen || !memory || !bitmap || !pixels) {
    if (bitmap) DeleteObject(bitmap);
    if (memory) DeleteDC(memory);
    if (screen) ReleaseDC(nullptr, screen);
    throw std::runtime_error("cannot initialize screen capture");
  }
  HGDIOBJ previous = SelectObject(memory, bitmap);
  const BOOL copied = BitBlt(memory, 0, 0, rect.width, rect.height, screen,
                             rect.x, rect.y, SRCCOPY | CAPTUREBLT);
  cv::Mat result;
  if (copied) {
    cv::Mat bgra(rect.height, rect.width, CV_8UC4, pixels);
    cv::cvtColor(bgra, result, cv::COLOR_BGRA2BGR);
  }
  SelectObject(memory, previous);
  DeleteObject(bitmap);
  DeleteDC(memory);
  ReleaseDC(nullptr, screen);
  if (!copied) throw std::runtime_error("screen capture failed");
  return result;
}

ScreenRect virtual_screen() {
  return {GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN),
          GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN)};
}

cv::Rect normalized_rect(POINT lhs, POINT rhs, int width, int height) {
  const int left = std::clamp(static_cast<int>(std::min(lhs.x, rhs.x)), 0, width);
  const int top = std::clamp(static_cast<int>(std::min(lhs.y, rhs.y)), 0, height);
  const int right = std::clamp(static_cast<int>(std::max(lhs.x, rhs.x)), 0, width);
  const int bottom = std::clamp(static_cast<int>(std::max(lhs.y, rhs.y)), 0, height);
  return {left, top, right - left, bottom - top};
}

cv::Scalar candidate_color(const uiparsercv::pipeline::UiElementCandidate& candidate) {
  using uiparsercv::pipeline::UiElementKind;
  if (candidate.kind == UiElementKind::ModelProposal) return {40, 200, 40};
  if (candidate.kind == UiElementKind::Icon) return {210, 120, 25};
  if (candidate.kind == UiElementKind::Text) return {25, 95, 230};
  return {190, 70, 190};
}

std::string compact(std::string value, std::size_t limit = 22) {
  value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
  if (value.size() > limit) value = value.substr(0, limit - 3) + "...";
  return value;
}

cv::Mat render_overlay(const cv::Mat& frame,
                       const uiparsercv::pipeline::PipelineResult& result) {
  cv::Mat overlay = frame.clone();
  for (const auto& association : result.associations) {
    if (association.proposal_index >= result.candidates.size() ||
        association.text_index >= result.candidates.size()) continue;
    const auto& proposal = result.candidates[association.proposal_index].box;
    const auto& text = result.candidates[association.text_index].box;
    cv::line(overlay,
             {static_cast<int>(proposal.x + proposal.width / 2),
              static_cast<int>(proposal.y + proposal.height / 2)},
             {static_cast<int>(text.x + text.width / 2),
              static_cast<int>(text.y + text.height / 2)},
             {230, 80, 220}, 1, cv::LINE_AA);
  }
  for (const auto& candidate : result.candidates) {
    const auto color = candidate_color(candidate);
    const cv::Rect rect{static_cast<int>(std::round(candidate.box.x)),
                        static_cast<int>(std::round(candidate.box.y)),
                        std::max(1, static_cast<int>(std::round(candidate.box.width))),
                        std::max(1, static_cast<int>(std::round(candidate.box.height)))};
    cv::rectangle(overlay, rect, color, 1, cv::LINE_AA);
    std::string label;
    if (candidate.kind == uiparsercv::pipeline::UiElementKind::ModelProposal) {
      label = candidate.model_class;
    } else if (candidate.kind == uiparsercv::pipeline::UiElementKind::Text) {
      label = compact(candidate.text);
    }
    if (!label.empty() && rect.width >= 55) {
      cv::putText(overlay, label, {rect.x + 2, std::max(11, rect.y + 11)},
                  cv::FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv::LINE_AA);
    }
  }
  return overlay;
}

void draw_mat(HDC dc, const cv::Mat& bgr, const RECT& destination) {
  if (bgr.empty()) return;
  cv::Mat bgra;
  cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
  BITMAPINFO info{};
  info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  info.bmiHeader.biWidth = bgra.cols;
  info.bmiHeader.biHeight = -bgra.rows;
  info.bmiHeader.biPlanes = 1;
  info.bmiHeader.biBitCount = 32;
  info.bmiHeader.biCompression = BI_RGB;
  SetStretchBltMode(dc, HALFTONE);
  StretchDIBits(dc, destination.left, destination.top,
                destination.right - destination.left, destination.bottom - destination.top,
                0, 0, bgra.cols, bgra.rows, bgra.data, &info, DIB_RGB_COLORS, SRCCOPY);
}

RECT fit_rect(const cv::Mat& image, RECT bounds) {
  if (image.empty()) return bounds;
  const double scale = std::min(
      static_cast<double>(bounds.right - bounds.left) / image.cols,
      static_cast<double>(bounds.bottom - bounds.top) / image.rows);
  const int width = static_cast<int>(image.cols * scale);
  const int height = static_cast<int>(image.rows * scale);
  const int x = bounds.left + ((bounds.right - bounds.left) - width) / 2;
  const int y = bounds.top + ((bounds.bottom - bounds.top) - height) / 2;
  return {x, y, x + width, y + height};
}

void begin_inference(HWND main_window, cv::Mat screenshot, ScreenRect region) {
  AppState* state = app_state(main_window);
  if (!state || state->processing) return;
  state->processing = true;
  state->region = region;
  state->overlay = std::move(screenshot);
  state->stats.reset();
  state->error.clear();
  SetWindowTextW(state->tree_json_field, L"Processing...");
  EnableWindow(state->new_snip_button, FALSE);
  InvalidateRect(main_window, nullptr, FALSE);
  auto runner = state->runner;
  cv::Mat input = state->overlay.clone();
  std::thread([main_window, runner, input = std::move(input), region]() mutable {
    auto* payload = new InferencePayload;
    payload->region = region;
    const auto start = std::chrono::steady_clock::now();
    try {
      const auto result = runner->run(input);
      payload->overlay = render_overlay(input, result);
      payload->stats = result.stats;
      std::ostringstream tree_json;
      uiparsercv::tree::write_json(tree_json, result.tree);
      payload->tree_json = tree_json.str();
    } catch (const std::exception& error) {
      payload->overlay = std::move(input);
      payload->error = error.what();
    }
    payload->elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (!PostMessageW(main_window, kInferenceDone, 0,
                      reinterpret_cast<LPARAM>(payload))) delete payload;
  }).detach();
}

void create_snip_window(HWND main_window) {
  const ScreenRect screen = virtual_screen();
  auto* state = new SnipState;
  state->main_window = main_window;
  state->screen = screen;
  try {
    state->snapshot = capture_screen(screen);
  } catch (...) {
    delete state;
    ShowWindow(main_window, SW_RESTORE);
    throw;
  }
  HWND window = CreateWindowExW(
      WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kSnipClass, L"New snip", WS_POPUP,
      screen.x, screen.y, screen.width, screen.height, nullptr, nullptr,
      GetModuleHandleW(nullptr), state);
  if (!window) {
    delete state;
    ShowWindow(main_window, SW_RESTORE);
    throw std::runtime_error("cannot create snipping overlay");
  }
  ShowWindow(window, SW_SHOW);
  SetForegroundWindow(window);
  SetCursor(LoadCursorW(nullptr, MAKEINTRESOURCEW(32515)));
}

LRESULT CALLBACK snip_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
  auto* state = reinterpret_cast<SnipState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
  if (message == WM_NCCREATE) {
    auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
    state = static_cast<SnipState*>(create->lpCreateParams);
    SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
  }
  switch (message) {
    case WM_LBUTTONDOWN:
      state->anchor = {GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
      state->cursor = state->anchor;
      state->dragging = true;
      SetCapture(window);
      return 0;
    case WM_MOUSEMOVE:
      if (state && state->dragging) {
        state->cursor = {GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
        InvalidateRect(window, nullptr, FALSE);
      }
      return 0;
    case WM_LBUTTONUP:
      if (state && state->dragging) {
        state->cursor = {GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
        state->dragging = false;
        ReleaseCapture();
        const cv::Rect selected = normalized_rect(
            state->anchor, state->cursor, state->snapshot.cols, state->snapshot.rows);
        if (selected.width >= 4 && selected.height >= 4) {
          cv::Mat screenshot = state->snapshot(selected).clone();
          const ScreenRect region{state->screen.x + selected.x,
                                  state->screen.y + selected.y,
                                  selected.width, selected.height};
          HWND main = state->main_window;
          DestroyWindow(window);
          ShowWindow(main, SW_RESTORE);
          SetForegroundWindow(main);
          begin_inference(main, std::move(screenshot), region);
        }
      }
      return 0;
    case WM_KEYDOWN:
      if (wparam == VK_ESCAPE && state) {
        HWND main = state->main_window;
        DestroyWindow(window);
        ShowWindow(main, SW_RESTORE);
        SetForegroundWindow(main);
      }
      return 0;
    case WM_PAINT: {
      PAINTSTRUCT paint{};
      HDC dc = BeginPaint(window, &paint);
      if (state) {
        cv::Mat dimmed;
        state->snapshot.convertTo(dimmed, -1, 0.48, 0.0);
        if (state->dragging) {
          const cv::Rect selected = normalized_rect(
              state->anchor, state->cursor, state->snapshot.cols, state->snapshot.rows);
          if (selected.width > 0 && selected.height > 0) {
            state->snapshot(selected).copyTo(dimmed(selected));
            cv::rectangle(dimmed, selected, {255, 255, 255}, 1, cv::LINE_AA);
          }
        }
        RECT client{};
        GetClientRect(window, &client);
        draw_mat(dc, dimmed, client);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255));
        TextOutW(dc, 24, 20, L"Drag to select a region   |   Esc to cancel", 43);
      }
      EndPaint(window, &paint);
      return 0;
    }
    case WM_DESTROY:
      delete state;
      SetWindowLongPtrW(window, GWLP_USERDATA, 0);
      return 0;
  }
  return DefWindowProcW(window, message, wparam, lparam);
}

void paint_main(HWND window, AppState& state) {
  PAINTSTRUCT paint{};
  HDC dc = BeginPaint(window, &paint);
  RECT client{};
  GetClientRect(window, &client);
  RECT image_area{0, 0, std::max(1L, client.right - kPanelWidth), client.bottom};
  HBRUSH image_background = CreateSolidBrush(RGB(238, 240, 243));
  FillRect(dc, &image_area, image_background);
  DeleteObject(image_background);
  RECT panel{image_area.right, 0, client.right, client.bottom};
  HBRUSH panel_background = CreateSolidBrush(RGB(248, 249, 251));
  FillRect(dc, &panel, panel_background);
  DeleteObject(panel_background);
  if (!state.overlay.empty()) draw_mat(dc, state.overlay, fit_rect(state.overlay, image_area));

  SetBkMode(dc, TRANSPARENT);
  SetTextColor(dc, RGB(35, 35, 38));
  int x = panel.left + 18;
  int y = 88;
  const std::wstring size = state.region.width > 0
      ? std::to_wstring(state.region.width) + L" x " + std::to_wstring(state.region.height)
      : L"No region selected";
  TextOutW(dc, x, y, size.c_str(), static_cast<int>(size.size()));
  y += 28;
  const wchar_t* status = state.processing ? L"Processing screenshot..."
      : (state.stats ? L"Overlay ready" : L"Click New snip to begin");
  TextOutW(dc, x, y, status, static_cast<int>(wcslen(status)));
  y += 36;
  TextOutW(dc, x, y, L"Legend", 6);
  struct Legend { COLORREF color; const wchar_t* text; } legends[] = {
      {RGB(40, 200, 40), L"UITag proposal"},
      {RGB(25, 120, 210), L"OmniParser support"},
      {RGB(230, 95, 25), L"OCR text"},
      {RGB(220, 80, 230), L"Model - OCR association"}};
  for (const auto& legend : legends) {
    y += 28;
    HPEN pen = CreatePen(PS_SOLID, 1, legend.color);
    HGDIOBJ old = SelectObject(dc, pen);
    Rectangle(dc, x, y - 10, x + 20, y + 4);
    SelectObject(dc, old);
    DeleteObject(pen);
    TextOutW(dc, x + 32, y - 11, legend.text, static_cast<int>(wcslen(legend.text)));
  }
  if (state.stats) {
    y += 42;
    TextOutW(dc, x, y, L"Detections", 10);
    const std::wstring counts = L"UITag " + std::to_wstring(state.stats->uitag_count) +
        L"   Icons " + std::to_wstring(state.stats->icon_count) +
        L"   OCR " + std::to_wstring(state.stats->text_region_count);
    y += 26;
    TextOutW(dc, x, y, counts.c_str(), static_cast<int>(counts.size()));
    const std::wstring timing = L"Inference " + std::to_wstring(state.elapsed_ms) + L" ms";
    y += 26;
    TextOutW(dc, x, y, timing.c_str(), static_cast<int>(timing.size()));
  }
  TextOutW(dc, x, 490, L"JSON tree", 9);
  if (!state.error.empty()) {
    std::wstring error(state.error.begin(), state.error.end());
    SetTextColor(dc, RGB(190, 40, 40));
    TextOutW(dc, x, client.bottom - 40, error.c_str(), static_cast<int>(error.size()));
  }
  EndPaint(window, &paint);
}

LRESULT CALLBACK main_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
  AppState* state = app_state(window);
  if (message == WM_NCCREATE) {
    auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
    state = static_cast<AppState*>(create->lpCreateParams);
    state->main_window = window;
    SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
  }
  switch (message) {
    case WM_CREATE:
      state->new_snip_button = CreateWindowW(
          L"BUTTON", L"New snip", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
          0, 0, 150, 42, window, reinterpret_cast<HMENU>(kNewSnipId),
          GetModuleHandleW(nullptr), nullptr);
      state->exit_button = CreateWindowW(
          L"BUTTON", L"Exit", WS_CHILD | WS_VISIBLE,
          0, 0, 72, 42, window, reinterpret_cast<HMENU>(kExitId),
          GetModuleHandleW(nullptr), nullptr);
      state->tree_json_field = CreateWindowExW(
          WS_EX_CLIENTEDGE, L"EDIT", L"{}",
          WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL |
              ES_READONLY,
          0, 0, 100, 100, window, nullptr, GetModuleHandleW(nullptr), nullptr);
      SendMessageW(state->tree_json_field, WM_SETFONT,
                   reinterpret_cast<WPARAM>(GetStockObject(ANSI_FIXED_FONT)), TRUE);
      return 0;
    case WM_SIZE: {
      RECT client{};
      GetClientRect(window, &client);
      const int panel_left = std::max(0L, client.right - kPanelWidth);
      MoveWindow(state->new_snip_button, panel_left + 18, 18, 190, 44, TRUE);
      MoveWindow(state->exit_button, panel_left + 220, 18, 82, 44, TRUE);
      MoveWindow(state->tree_json_field, panel_left + 18, 510,
                 kPanelWidth - 36, std::max(80L, client.bottom - 528), TRUE);
      InvalidateRect(window, nullptr, FALSE);
      return 0;
    }
    case WM_COMMAND:
      if (LOWORD(wparam) == kNewSnipId && !state->processing) {
        ShowWindow(window, SW_MINIMIZE);
        SetTimer(window, kSnipTimer, 260, nullptr);
      } else if (LOWORD(wparam) == kExitId) {
        DestroyWindow(window);
      }
      return 0;
    case WM_TIMER:
      if (wparam == kSnipTimer) {
        KillTimer(window, kSnipTimer);
        create_snip_window(window);
      }
      return 0;
    case kInferenceDone: {
      std::unique_ptr<InferencePayload> payload(
          reinterpret_cast<InferencePayload*>(lparam));
      state->overlay = std::move(payload->overlay);
      state->stats = payload->stats;
      state->region = payload->region;
      state->elapsed_ms = payload->elapsed_ms;
      state->error = std::move(payload->error);
      const std::wstring tree_json = utf8_to_wide(payload->tree_json);
      SetWindowTextW(state->tree_json_field,
                     tree_json.empty() ? L"{}" : tree_json.c_str());
      state->processing = false;
      EnableWindow(state->new_snip_button, TRUE);
      InvalidateRect(window, nullptr, FALSE);
      return 0;
    }
    case WM_PAINT:
      paint_main(window, *state);
      return 0;
    case WM_CLOSE:
      DestroyWindow(window);
      return 0;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(window, message, wparam, lparam);
}

int run(HINSTANCE instance) {
  SetProcessDPIAware();
  WNDCLASSW main_class{};
  main_class.lpfnWndProc = main_proc;
  main_class.hInstance = instance;
  main_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  main_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
  main_class.lpszClassName = kMainClass;
  if (!RegisterClassW(&main_class)) throw std::runtime_error("cannot register main window");

  WNDCLASSW snip_class{};
  snip_class.lpfnWndProc = snip_proc;
  snip_class.hInstance = instance;
  snip_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32515));
  snip_class.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
  snip_class.lpszClassName = kSnipClass;
  if (!RegisterClassW(&snip_class)) throw std::runtime_error("cannot register snip window");

  uiparsercv::pipeline::PipelineOptions options;
  options.uitag_model_path = UIPARSERCV_UITAG_MODEL;
  options.icon.model_path = UIPARSERCV_ICON_MODEL;
  options.ocr_detector.model_path = UIPARSERCV_OCR_DET_MODEL;
  options.ocr_recognizer.model_path = UIPARSERCV_OCR_REC_MODEL;
  options.ocr_recognizer.config_path = UIPARSERCV_OCR_REC_CONFIG;
  AppState state;
  state.runner = std::make_shared<uiparsercv::pipeline::PipelineRunner>(options);

  HWND window = CreateWindowExW(
      0, kMainClass, kTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, CW_USEDEFAULT, 1180, 720, nullptr, nullptr, instance, &state);
  if (!window) throw std::runtime_error("cannot create main window");
  ShowWindow(window, SW_SHOW);
  UpdateWindow(window);

  MSG message{};
  while (GetMessageW(&message, nullptr, 0, 0) > 0) {
    TranslateMessage(&message);
    DispatchMessageW(&message);
  }
  return static_cast<int>(message.wParam);
}

}  // namespace

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int) {
  try {
    return run(instance);
  } catch (const std::exception& error) {
    MessageBoxA(nullptr, error.what(), "live-overlay error", MB_OK | MB_ICONERROR);
    return 1;
  }
}
