# Depth Anything V2 Report

Repository for the DLVC Assignment 3 experiment using Depth Anything V2.

## Current dataset

The custom photo dataset is organized by challenge category:

```text
data/own_photos/
  indoor/
  outdoor/
  mirror/
  glass_window/
  shiny_metal/
  plain_wall/
  very_close_object/
  dark_image/
```

Current status:

| Category | Status |
|---|---|
| Indoor | ready |
| Outdoor | ready |
| Mirror | ready |
| Glass / Window | ready |
| Shiny Metal | ready |
| Plain Wall | ready |
| Very Close Object | ready |
| Dark Image | still needs additional photos |

## Planned experiment

1. Run Depth Anything V2 on all images.
2. Save raw depth arrays, colored depth maps, and side-by-side visualizations.
3. Perform qualitative analysis for each category.
4. Perform a small ordinal depth test using manually selected close/far point pairs.
5. Discuss success and failure cases in the final report.

## Important metadata files

- `data/metadata/own_photos_metadata.csv`: category labels and scene descriptions.
- `data/metadata/manual_ordinal_pairs_template.csv`: template for manual ordinal depth evaluation.

## Results folder

Generated outputs should be saved under:

```text
results/
  depth_maps/
  visualizations/
  tables/
```

