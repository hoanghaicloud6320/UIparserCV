# UIparserCV

UIparserCV là dự án C++20 để phân tích ảnh chụp giao diện người dùng và xuất ra cây cấu trúc UI.

Pipeline mục tiêu:

```text
ảnh input
  -> icon detector
  -> OCR detector + recognizer
  -> thuật toán merge box / containment tree bằng C++
  -> UI tree: JSON, XML, YAML, ...
```

## Tech Stack

- C++20
- GCC từ MSYS2 UCRT64, dùng `g++`
- CMake
- OpenCV nếu cần cho đọc ảnh, tiền xử lý, debug trực quan
- Icon detection: model icon detector từ Microsoft OmniParser, ưu tiên format ONNX
- OCR: PP-OCRv6 tiny/small `det.onnx` và `rec.onnx`

Dependency sẽ ưu tiên cài bằng MSYS2 `pacman` nếu có package phù hợp. Nếu `pacman` không có, dependency có thể được kéo local vào `vendor/`, `external/`, hoặc một thư mục tương tự sau khi được duyệt.

## Trạng Thái Hiện Tại

Repo đang ở giai đoạn dựng nền và thử từng component độc lập trước khi ráp pipeline.

Đã có:

- CMake project C++20.
- Core module đầu tiên cho box containment tree.
- Inference module C++ cho ONNX Runtime + OpenCV, tách riêng khỏi core tree.
- Wrapper độc lập cho icon detector, OCR detector, OCR recognizer.
- Probe kiểm tra thuật toán tree và xuất JSON mẫu.
- Probe kiểm tra đường dẫn model.
- Probe kiểm tra OpenCV đọc ảnh.
- Probe chạy các component inference trên ảnh smoke test.

Đã kiểm tra local:

- `g++` hoạt động.
- `cmake` hoạt động.
- OpenCV được tìm thấy qua `pkg-config opencv4`.
- ONNX Runtime được tìm thấy qua `pkg-config libonnxruntime`, package MSYS2 UCRT64 hiện là `1.26.0`.
- `ui_tree_probe` pass qua CTest.
- `component_probe` chạy icon detector, OCR detector, OCR recognizer trên ảnh test tự tạo.
- `onnxruntime_probe` khởi tạo được ONNX Runtime và load được một ONNX model nhỏ tạo tạm trong `build/tmp`.
- PP-OCRv6 tiny ONNX det/rec đã được tải về `models/ocr` từ nguồn official PaddleOCR/Paddle model ecology và chạy dummy inference thành công.
- Icon detector ONNX đã được đặt ở `models/icon_detect/model.onnx` và chạy dummy inference thành công.
- Icon detector defaults follow Microsoft OmniParser Gradio defaults: confidence threshold `0.05`, NMS/IoU threshold `0.1`, image size `640`.
- `opencv_probe` đọc được ảnh mẫu 2x2.
- `model_file_probe` pass với đủ icon model, OCR det model, OCR rec model.

## Layout Model Dự Kiến

Mặc định project tìm model ở:

```text
models/
  icon_detect/
    model.onnx
  ocr/
    det.onnx
    rec.onnx
```

Có thể override khi configure:

```powershell
cmake -S . -B build `
  -DUIPARSERCV_ICON_MODEL=C:/path/to/icon_detect/model.onnx `
  -DUIPARSERCV_OCR_DET_MODEL=C:/path/to/det.onnx `
  -DUIPARSERCV_OCR_REC_MODEL=C:/path/to/rec.onnx
```

Model lớn không được commit vào Git.

## Build

Trong shell có MSYS2 UCRT64 toolchain trên PATH:

```powershell
cmake -S . -B build
cmake --build build
```

Nếu muốn chỉ định Ninja:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

## Chạy Kiểm Thử

```powershell
ctest --test-dir build --output-on-failure
```

Hiện tại CTest chạy `ui_tree_probe`, kiểm tra quan hệ chứa nhau giữa các UI box mẫu.
Khi OpenCV và ONNX Runtime có sẵn, CTest cũng chạy `component_probe` để kiểm tra các wrapper inference.

## Component Probes

### Tree Merge Probe

Không cần OpenCV hay model:

```powershell
.\build\tools\ui_tree_probe.exe
```

Probe tạo vài box mẫu, dựng containment tree, kiểm tra cấu trúc, rồi in JSON.

### ONNX Runtime Probe

Kiểm tra ONNX Runtime có load được model không:

```powershell
.\build\tools\onnxruntime_probe.exe path\to\model.onnx
```

Probe có thể nhận nhiều model cùng lúc, ví dụ cặp OCR:

```powershell
.\build\tools\onnxruntime_probe.exe models\ocr\det.onnx models\ocr\rec.onnx
```

Chạy inference smoke test bằng tensor zero:

```powershell
.\build\tools\onnxruntime_probe.exe --run-dummy models\ocr\det.onnx models\ocr\rec.onnx
```

Nếu chạy không kèm model path, probe chỉ in ONNX Runtime API version.

Trên Windows có thể có `onnxruntime.dll` khác trong `C:\Windows\System32`. Dev build hiện copy DLL của MSYS2 UCRT64 vào cạnh `onnxruntime_probe.exe` để Windows loader dùng đúng bản runtime.

### Model File Probe

Kiểm tra model files đã nằm đúng chỗ chưa:

```powershell
.\build\tools\model_file_probe.exe
```

Probe này chưa chạy inference. Nó chỉ xác nhận wiring đường dẫn model.

### OpenCV Probe

Được build khi CMake tìm thấy OpenCV:

```powershell
.\build\tools\opencv_probe.exe path\to\screenshot.png
```

Probe đọc ảnh và in metadata cơ bản như width, height, channels, depth.

### Component Probe

Chạy icon detector, OCR detector, và OCR recognizer qua API wrapper C++:

```powershell
.\build\tools\component_probe.exe
```

Có thể truyền ảnh thật:

```powershell
.\build\tools\component_probe.exe path\to\screenshot.png
```

## Module Dự Kiến

```text
src/
  image/        đọc ảnh và tiền xử lý
  detect/       tích hợp icon detector
  ocr/          tích hợp PP-OCRv6 detection / recognition
  infer/        wrapper ONNX Runtime dùng chung
  tree/         merge box và containment tree bằng C++
  export/       JSON / XML / YAML
  app/          command-line entry points
```

Nguyên tắc triển khai là giữ từng module chạy được độc lập trước, rồi mới ráp thành pipeline hoàn chỉnh.

## Pipeline Shape

Pipeline chính nên chỉ điều phối các component, không chứa chi tiết preprocess/postprocess:

```mermaid
flowchart LR
  A["input image"] --> B["IconDetector"]
  A --> C["OcrTextDetector"]
  C --> D["crop text regions"]
  D --> E["OcrTextRecognizer"]
  B --> F["UiElementCandidate merge"]
  E --> F
  C --> F
  F --> G["containment tree"]
  G --> H["JSON / XML / YAML export"]
```

Schema trung gian hiện là `UiElementCandidate`: `kind`, `box`, `detection_score`, `text`, `text_confidence`, `interactive`, `source`. Kiểu này đủ đơn giản để debug bằng log/JSON, nhưng vẫn mở rộng được khi thêm caption, role, hoặc state.

## Các Điểm Cần Chốt

- OCR DB postprocess hiện bám PaddleOCR C++ flow và dùng Clipper offset cho `unclip`.
- Candidate merge bám logic OmniParser `remove_overlap_new`: text boxes được ưu tiên, icon hấp thụ text nằm bên trong, icon nằm trong text bị bỏ.
- Format export đầu tiên: JSON nên đi trước vì dễ validate và dễ dùng cho tooling. XML/YAML có thể thêm sau khi schema UI tree ổn định.

## Model Sources

- PP-OCRv6 tiny ONNX detection: `PP-OCRv6_tiny_det_onnx_infer.tar`
- PP-OCRv6 tiny ONNX recognition: `PP-OCRv6_tiny_rec_onnx_infer.tar`

Các link này được PaddleOCR docs liệt kê trong phần Android deployment cho PP-OCRv6 tiny/small ONNX.
