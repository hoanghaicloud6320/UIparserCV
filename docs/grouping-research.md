# Model-first UI Parsing Handoff

This document supersedes the heuristic-grouping research plan. The previous
segmentation experiments were useful for exposing failure modes, but they are
not the intended primary detection architecture going forward.

## Architecture decision

Use the UITag YOLO11 UI detector as the primary source of UI element boxes.
Do not continue growing the hand-written line, color, contour, or proximity
heuristics into a competing general-purpose segmenter.

The reusable C++ detector is implemented in `src/detect/uitag_detector.cpp`,
with `tools/uitag_yolo_probe.cpp` retained as an independent debug executable.
It currently runs independently from the production pipeline and implements:

- ONNX Runtime inference without Python or Ultralytics at runtime;
- 640 x 640 tiled inference with 20% overlap;
- edge-anchored tiles matching the original UITag implementation;
- YOLO11 output decoding for nine UI classes;
- per-tile filtering/NMS and class-agnostic cross-tile NMS;
- full-image coordinate mapping;
- raw JSON and debug-overlay output.

The standalone probe has been smoke-tested on finance, ecommerce, and weather
screenshots. Desktop results contain useful element candidates. The mobile
weather image produces mostly `Unknown`, which is consistent with the model's
desktop-oriented training domain.

## Target pipeline

```text
Screenshot
  -> UITag tiled detector (primary element proposals)
  -> OCR detector + recognizer (text content and text boxes)
  -> proposal normalization and deduplication
  -> conservative relation/grouping stage
  -> strict immediate-containment tree
  -> JSON + debug overlay
```

### UITag detector responsibilities

- Find atomic controls and useful UI regions.
- Preserve the model class, confidence, and original box.
- Preserve raw detections before any merging.
- Produce deterministic tiled results for the same image and thresholds.

The model class is evidence, not ground truth. Initial overlays show useful
boxes with noisy labels, especially among `Button`, `Input_Elements`,
`Information_Display`, and `Unknown`.

### OCR responsibilities

- Recognize text independently of the model class.
- Keep tight OCR geometry; never replace a text box with a larger model box.
- Associate text with a model proposal only during relation/grouping.
- Permit OCR-only elements where the model misses visible text.

### Normalization responsibilities

- Remove cross-tile duplicates without hiding the raw detections.
- Clamp coordinates and reject invalid boxes.
- Mark near-identical OCR/model boxes as associated observations rather than
  blindly deleting one source.
- Keep source provenance and confidence in exported debug data.

### Grouping responsibilities

Grouping becomes a conservative relation layer over model and OCR proposals,
not an image segmenter.

Start with:

1. strict immediate containment;
2. text-to-control association when containment or proximity is unambiguous;
3. repeated sibling structure only when supported by multiple model boxes;
4. abstention when several layouts remain plausible.

The selected hierarchy must remain laminar: one immediate parent per node, no
cycles, no transitive edges, and no partially overlapping siblings.

## Role of the old heuristics

The existing visual-container, line-rectangle, and boundaryless grouping code
must not automatically feed the default pipeline after model integration.
Keep it temporarily for comparison and diagnostics.

Later, an individual heuristic may return only as bounded supporting evidence,
for example:

- suggesting a container around several already-detected model elements;
- splitting an obviously oversized model proposal;
- recovering a candidate when both model and OCR have a documented miss;
- increasing or decreasing a relation score without creating a box itself.

Any such use needs an ablation against the model-only baseline. A heuristic is
accepted only when it improves a named failure class without causing broad
false-positive growth.

## Integration sequence

### Phase 1: detector component (complete)

- [x] Move the tested probe logic into a reusable `UitagDetector` component.
- [x] Keep the standalone probe as a regression/debug executable.
- [x] Add unit tests for tile positions, output decoding, coordinate restoration,
  and cross-tile NMS.
- [x] Record inference options and model metadata in raw output.

### Phase 2: model-only exper1 baseline

- Run all exper1 images with unchanged thresholds.
- Save raw model detections and overlays separately from every heuristic run.
- Manually inspect desktop and mobile results.
- Catalogue duplicate boxes, oversized boxes, missed controls, class noise,
  and domain-shift failures.

### Phase 3: OCR association

- Combine raw OCR and model observations without constructing semantic groups.
- Visualize both boxes and their association edges.
- Verify that tight text boxes remain intact, including the prior weather-card
  failure where text geometry was confused with a larger card box.

### Phase 4: conservative tree

- Build immediate containment from normalized candidates.
- Add only high-confidence text/control associations.
- Compare against model-only and the committed heuristic baseline.

### Phase 5: optional support evidence

- Revisit line/color/proximity evidence only for specific catalogued misses.
- Require a measurable improvement and keep every support feature removable.

## Immediate next task

Do not integrate the detector directly into `PipelineRunner` yet. Produce a
complete model-only exper1 run next. That run becomes the baseline for deciding
merge and grouping behavior.

## Historical findings retained

- Strict immediate containment is deterministic and remains useful downstream.
- Tight OCR boxes must survive even inside larger icon/card/model boxes.
- Closed contours and four supported lines frequently generate both valid UI
  regions and convincing false positives.
- Boundaryless alignment/proximity can describe real groups, but it also admits
  several plausible layouts and should not be a primary detector.
- Texture alone cannot reliably distinguish photographs from dense UI panels.
- A visually closed region is neither necessary nor sufficient for a semantic
  container.
