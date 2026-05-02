# Multi-Scale Strategy — Figure Description

This file describes the crop-strategy diagram for slide 4 ("Our idea: multi-scale")
and the method section of the report.
Suggested tools: Excalidraw or draw.io (a spatial diagram works better than Mermaid here).

## What the diagram should show

A single input image split into a 3-level pyramid:

```
Original image (480×480 after resize)
│
├── Scale 1.0 — full image → MaskCut → masks M₁
│
├── Scale 0.75 — sliding windows at 0.75× → MaskCut on each crop
│   ├── crop (0,0)–(360,360) → masks projected back → M₂
│   ├── crop (120,0)–(480,360) → masks projected back → M₃
│   └── ... (overlap = 0.3)
│
└── Scale 0.5 — sliding windows at 0.5×  → MaskCut on each crop
    ├── crop (0,0)–(240,240) → masks → M₄
    └── ...
                    │
                    ▼
           Merge all Mᵢ via IoU-NMS
           (graph-based: union connected components)
                    │
                    ▼
           Final pseudo-mask set for this image
```

## Key design choices to label on the figure

| Choice | Value | Why |
|---|---|---|
| Crop overlap | 0.3 | Ensures objects near window edges are captured |
| Scales | 1.0, 0.75, 0.5 | Covers 1×–4× scale range; 0.25× too extreme for 480px input |
| MaskCut N per crop | 3 | Same as baseline — locked param |
| NMS IoU threshold | 0.5 | Standard; above this, two masks are near-duplicates |
| Containment threshold | 0.7 | Suppress a mask if 70% of it is inside a larger one |

## Crop back-projection

Each crop window `[x1, y1, x2, y2]` in original-image coordinates.
A mask `m` produced by MaskCut on the resized crop is mapped back:

```
mask_x_orig = x1 + mask_x_crop * (x2 - x1) / fixed_size
mask_y_orig = y1 + mask_y_crop * (y2 - y1) / fixed_size
```

This is the key correction vs the earlier broken implementation (which cropped from the
already-resized image, losing resolution for small objects).

## Figure caption draft

> **Figure 2.** Multi-scale MaskCut strategy. Sliding windows at 0.75× and 0.5× zoom are
> cropped from the original image, resized to 480×480 for inference, and projected back to
> original coordinates. Proposals from all scales are merged via graph-based IoU-NMS,
> recovering small objects missed at full scale.

## TODO

- [ ] Draw this in Excalidraw or draw.io — use the ASCII art above as the layout guide.
- [ ] Export as SVG/PNG and add to `02_visualizations/selected/` or here.
