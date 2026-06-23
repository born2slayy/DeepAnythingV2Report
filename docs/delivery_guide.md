Here is summary
photos/ contains the input photos:
- asaken/ = my own photos
- indoor, outdoor, mirror, glass_window, shiny_metal, plain_wall, very_close_object = categorized test cases

results/ contains all generated outputs:
- small/
- base/
- large/
- results.csv = all per-image/per-model measurements
- summary_by_model.csv = average runtime per model

We ran Depth Anything V2 with three checkpoints:
- Small: vits
- Base: vitb
- Large: vitl

All runs used CUDA/GPU.

Average inference time:
- Small: 0.229098 sec/image
- Base: 0.236437 sec/image
- Large: 0.424743 sec/image

There is no real ground-truth depth, so we should not claim real accuracy. The results are mainly qualitative depth maps plus runtime comparison.