# UITag model-only exper1 baseline

Date: 2026-07-12

## Run configuration

- Model: `models/uitag/uitag-yolo11s-ui-detect-v1.onnx`
- Tile size / step: `640 / 512` (20% overlap)
- Confidence threshold: `0.25`
- Per-tile class-aware NMS IoU: `0.70`
- Cross-tile class-agnostic NMS IoU: `0.50`
- Inputs: all 20 images under `build/exper1/inputs`
- Artifacts: `build/exper1/uitag_model_only/*.raw.json` and
  `build/exper1/uitag_model_only/*.overlay.png`

Each JSON contains both `raw_detections` (after per-tile NMS, before cross-tile
NMS) and final `detections`, plus the complete inference options and model path.

## Counts

| Images | Raw | Final | Removed by cross-tile NMS |
|---|---:|---:|---:|
| Finance desktop | 65 | 49 | 16 |
| Mobile weather | 60 | 43 | 17 |
| Ecommerce desktop | 57 | 37 | 20 |
| 17 synthetic images | 132 | 92 | 40 |
| **Total** | **314** | **221** | **93** |

The 17 synthetic images yield between 2 and 8 final detections each. Repeated
layouts do not receive stable classes: most boxes are `Unknown`; similar cards
alternate between `Button`, `Information_Display`, `Others`, and `Unknown`.

## Manual inspection catalogue

### Finance desktop

- Useful: sidebar items, summary cards, table cells, filters, and primary action
  controls are mostly proposed with reasonably tight boxes.
- Duplicate/overlap: tiling produces 16 removable duplicates. The large balance
  card also has several overlapping proposals describing the card and its inner
  regions; these are semantic alternatives rather than exact duplicates.
- Oversized: no final box exceeds 15% of the image, but some column/table boxes
  span several rows and should not be treated as atomic controls.
- Misses: some small icons and plain text (for example logout and chart labels)
  are not proposed. OCR must supply these.
- Class noise: visually similar sidebar entries alternate among `Menu`,
  `Input_Elements`, and `Unknown`; table values are often
  `Information_Display`, but some columns become `Input_Elements`.

### Mobile weather

- Domain shift is severe: 39 of 43 final boxes are `Unknown`; the remaining four
  are `Others`. No useful semantic class survives.
- The detector does find many headings, hourly cells, daily rows, and footer
  metric cards, but geometry is inconsistent: some boxes cover text only, some
  cover whole rows, and several hourly boxes partially overlap.
- Large visual content such as the sun illustration is missed. The hamburger
  control is also missed while the search icon is boxed as `Unknown`.
- Cross-tile NMS removes 17 duplicates, but it cannot resolve competing nested
  or partially overlapping layout interpretations.

### Ecommerce desktop

- Useful: product actions, thumbnails, feature rows, search/account/cart
  controls, tabs, and specification cards are detected well enough to seed a
  relation stage.
- Duplicate/oversized: the main product image has both a large
  `Input_Elements` proposal and a nested `Visual_Elements` proposal. The larger
  one is the only final detection in the full run exceeding 15% image area.
- Misses: breadcrumb text and some small icon/text observations are absent and
  must come from OCR.
- Class noise: descriptive feature rows are frequently classified as `Button`;
  tabs alternate between `Button` and `Information_Display`; the product image
  is labelled as both input and visual evidence.

### Synthetic images

- The model generally recognizes outlined input fields, buttons, and repeated
  cards, but performance depends strongly on layout and tile position.
- Repeated siblings are not reliable evidence by class alone. In otherwise
  identical grids, one card can be missed while its siblings alternate between
  `Button` and `Unknown`.
- Text-only headings are commonly absent, which is acceptable only if OCR-only
  candidates remain enabled.
- Narrow portrait images use two heavily overlapping edge-anchored tiles; their
  raw count can be more than twice the final count. This is expected tiling
  duplication and is handled by cross-tile NMS.

## Baseline conclusions

1. Desktop results are useful as primary proposals, but model classes must stay
   soft evidence rather than roles.
2. Mobile and synthetic layouts demonstrate clear domain shift; `Unknown` must
   remain a valid proposal rather than being discarded.
3. Keep raw detections because NMS alone cannot distinguish a duplicate from a
   legitimate nested/alternative region.
4. The next stage should associate independent OCR boxes without merging their
   geometry into model boxes. Large/ambiguous model boxes should be retained as
   observations until the conservative relation stage.
