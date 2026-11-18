# Measurement bench

How close are the drawn rooms to the real ones? This bench answers that with exact ground truth, so changes to the
scale, the vectorizer or the models can be compared with numbers instead of by eye.

## What is in it

- `gen.py` renders synthetic marketing plans (same layouts, furniture and styles as `zmc/synth.py`, at 1600 px) and
  prints room names and **clear dimensions** on them the way brochures do (`2.69 x 2.20`, `2,69 x 2,20`,
  `12'4" x 11'0"`), sometimes the declared area too. The truth (clear spans face to face of walls, net area, footprint)
  is measured on the supersampled raster. Variants:
  - `base`: horizontal text, 1600 px
  - `small`: 900 px (screenshot / web brochure)
  - `vertical`: dimensions rotated 90° in narrow rooms
  - `photo`: phone photo of a printed brochure (perspective, rotation, uneven light, blur, JPEG)
  - `stretch`: image stretched 5–12 % in X
- `core.py` runs the pipeline up to the OCR once (models + OCR are the slow part, cached in `<folder>/_cache`) and then
  tries cheap variants on the same reading: with or without the local fit, and **leaving one printed dimension out**
  (the room is then measured as if the brochure had not printed it).
- `run.py` loops over a folder and writes a JSON; `summary.py` prints the table.

Measured per room, on the final vector model: clear span in X and Y (median of rows / columns at 30, 50 and 70 % of the
room, face to face of walls and openings) and net area. A room of the truth that lands in the same model room as
another is counted as a topology error (merged), not as a measurement.

```bash
python bench/measure/gen.py /tmp/zmc_bench 8 5000          # 8 plans per variant, seeds 5000..5007
python bench/measure/run.py /tmp/zmc_bench results.json      # configs v03 and adj (see run.py)
python bench/measure/summary.py results.json
```

Real brochures can be added with a hand-written JSON next to the image (same fields: room centres in image pixels and
clear spans in metres). Developer images are for evaluation only and stay out of the repository.

## Results

See [RESULTS.md](RESULTS.md).
