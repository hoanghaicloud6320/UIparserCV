# UI Grouping Research Notes

This is a living draft for grouping experiments. Keep hypotheses, failures, and
possible next steps here so later attempts do not erase earlier reasoning.

## Stable baseline

- Preserve raw OCR boxes even when they are inside an icon/card box.
- Build the final tree with strict immediate containment only.
- Equal rectangles remain siblings.
- Visual/color container detection is experimental and disabled by default.

## Problem statement

The advanced problem is not simply "find rectangles". It is to propose and
rank a laminar hierarchy of UI regions from incomplete and ambiguous evidence.
Multiple row/column decompositions can be geometrically plausible.

### Hard constraints for the main tree

- No cycles.
- A node has one immediate parent.
- Parent strictly contains child.
- Siblings must not partially overlap.
- Do not store transitive edges.
- Do not cut through high-confidence atomic OCR/icon boxes without strong
  contrary evidence.
- Abstaining is valid when evidence is weak.

### Positive evidence

- Explicit geometric containment.
- Boundary line support, corners, and T-junctions.
- Relative proximity: internal gap is clearly smaller than external gap.
- Horizontal/vertical/baseline alignment.
- Repeated geometry and repeated internal child layout.
- Negative-space gutters.
- Inside/outside color difference and locally homogeneous backgrounds.
- Stability across scale, blur, and edge thresholds.
- Reading order (weak evidence only).

### Negative evidence

- Partial overlap with stronger regions.
- Photograph/high-texture interiors.
- Letter contours and decorative outlines.
- A hypothesis supported only by a fixed pixel threshold.
- A group that adds complexity without explaining repeated structure.

## Attempts

### Baseline: strict containment

Status: accepted. Deterministic and explainable.

### Visual closed-contour containers

Status: retained as an experimental feature, disabled by default.

Useful on flat weather cards and input fields. False positives occur on large
letters and product photographs. The general problem should operate on line
arrangements rather than require closed contours.

### Attempt 1: conservative boundaryless repeated groups

Scope:

- Repeated horizontal icon-text pairs.
- Repeated vertical primary-secondary text stacks.
- Relative gaps and alignment, not fixed pixels alone.
- Do not regroup elements already explained by an existing containing box.
- Require at least two geometrically similar instances.
- Reject member reuse and partial-overlap conflicts.
- Emit inferred group candidates with explicit source/confidence for debug.

Non-goals:

- Large section grouping.
- Toolbar-wide grouping.
- Choosing row versus column decompositions in tables.
- Semantic inference from recognized text.
- Competing hypotheses in the main tree.

### Attempt 1.5: four-side line rectangle proposals

Scope:

- Keep Attempt 1 unchanged.
- Extract horizontal/vertical line segments from per-channel color edges.
- Merge near-collinear segments and bridge small occlusion gaps.
- Generate only rectangles with support on all four sides.
- Reject very small/large rectangles and high-texture interiors.
- Deduplicate against existing detector boxes and other proposals.
- Feed accepted proposals into strict containment before boundaryless fallback.

Deferred:

- Three-side rectangles with an inferred missing edge.
- T-junction/minimal-face extraction.
- Proposal competition and global laminar selection.

Observed on `exper1`:

- Initial four-side enumeration produced too many composite rectangles.
- Minimal-proposal filtering reduced weather from 41 proposals to 9 and
  ecommerce from 23 to 4.
- Strong examples: weather rows/cards, ecommerce price/action panels,
  synthetic input fields and buttons.
- Known ambiguity: a proposal can join adjacent hourly cards when their shared
  or outer edges are stronger than the missing internal face evidence.
- One finance sidebar proposal extends beyond the visually intended item,
  showing that four supported lines alone do not guarantee the desired cell.
- Attempt 1 boundaryless groups remain useful after line rectangles: weather
  metric stacks and ecommerce feature text stacks still survive where no line
  rectangle explains them.
- Evidence precedence matters: exact closed-contour cells outrank composite
  line rectangles that contain or partially cross them.
- Weather reaches six exact hourly cards after retaining closed-contour child
  cells and suppressing conflicting line proposals.
- Photo suppression must combine texture with low atomic-content count; using
  texture alone misclassifies information-dense UI panels as photographs.

## Might try later

- Build a horizontal/vertical line arrangement with merged collinear segments.
- Use T-junctions and minimal planar faces; infer missing edges from continuation.
- Score repeated templates jointly instead of greedily selecting pairs.
- Detect photographic/high-texture exclusion zones.
- Reclassify large YOLO icon boxes with many distributed children as containers.
- Preserve competing row/column hypotheses outside the selected tree.
- Use a minimum-description-length/complexity penalty.
- Measure proposal stability across image scale and detector thresholds.
- Create manual annotations for acceptable alternative hierarchies, not one
  artificially unique ground-truth tree.
