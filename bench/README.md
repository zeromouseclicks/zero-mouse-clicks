# Furniture bench

How well does Zero Mouse Clicks read the furniture and fixtures of a brochure? This folder answers with numbers instead of impressions.

- `labels/*.json`: pieces marked by hand on real and example plans: type, box (pixels of the original image), and where the front faces (`N`, `S`, `E`, `W` in the image; `null` when it does not apply).
- `eval.py`: runs the full pipeline on each labelled plan and reports, per type, whether the model saw the piece, whether it was drawn in place with the right type, whether it faces the right way, and the size error of the drawn block. Writes `results.md`, `results.json` and one overlay per plan (green = label, red = drawn).
- `prelabel.py`: writes the pipeline's own pieces as a label file, to correct instead of starting from zero.
- `label.html`: a small page to draw and correct labels in the browser (open it from disk; nothing is uploaded).
- `RESULTS.md`: the latest run.

```bash
python bench/eval.py --images path/to/developer/plans --out bench/results
python bench/prelabel.py my_plan.jpg --area 51 -o bench/labels/my_plan.json   # then fix it in label.html
```

**Images from developers are not in this repository.** They belong to their owners: the labels keep only the file name and coordinates, and the images stay on your own disk (`bench/images/` is ignored by git). They are used to *measure*, never to train. The example plan (`docs/img/brochure.jpg`) is the project's own render and is included.

Types: `bed, nightstand, desk, chair, armchair, sofa, coffee_table, side_table, tv_unit, dining_table, stool, plant, fridge, cooktop, sink, washer, toilet, lavatory, shower, bathtub`.
