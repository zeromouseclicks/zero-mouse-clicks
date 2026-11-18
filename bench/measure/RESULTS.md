# Measurement bench — results (zmc 0.4.0, 2026-11-18)

40 synthetic brochures (8 per variant, seeds 5000–5007) plus the Blender example brochure of the website.
`v03` = zmc 0.3.0 as published. `adj` = 0.4.0: one thickness per wall line + walls fitted to the printed dimensions.
Errors are absolute; "median / p90"; spans in cm, areas in %. Only plans that got a scale are measured.

| variant | config | plans | scaled | OCR dims ok | rooms (merged) | span err, rooms with printed dims | span err, other rooms | leave-one-out | room area err % |
|---|---|---|---|---|---|---|---|---|---|
| base | v03 | 8 | 5 | 22/32 | 24 (0) | 3.0 / 5.6 | 3.6 / 7.4 | 3.3 / 5.7 | 1.8 / 3.5 |
| base | **adj** | 8 | 5 | 22/32 | 24 (0) | **0.6 / 1.5** | 2.2 / 6.3 | 1.2 / 4.3 | **0.4 / 2.0** |
| small | v03 | 8 | 4 | 13/32 | 20 (0) | 1.2 / 4.3 | 0.8 / 3.4 | 0.8 / 5.6 | 1.1 / 2.4 |
| small | **adj** | 8 | 4 | 13/32 | 20 (0) | **0.7 / 3.5** | 0.5 / 2.0 | 2.2 / 5.3 | **0.6 / 2.0** |
| stretch | v03 | 8 | 5 | 17/32 | 24 (0) | 2.0 / 6.7 | 4.2 / 7.5 | 2.9 / 8.1 | 1.7 / 2.9 |
| stretch | **adj** | 8 | 5 | 17/32 | 24 (0) | **0.6 / 3.0** | 3.3 / 5.2 | 3.1 / 8.3 | **0.5 / 2.2** |
| vertical | v03 | 8 | 7 | 14/29 | 32 (4) | 3.5 / 241 * | 6.6 / 299 * | 3.1 / 5.9 | 1.7 / 84 * |
| vertical | **adj** | 8 | 7 | 14/29 | 32 (4) | **0.8 / 241 *** | 6.3 / 299 * | 1.2 / 6.3 | **0.7 / 84 *** |
| photo | v03 | 8 | 3 | 16/32 | 15 (7) | 8.2 / 25.9 | 16.5 / 18.1 | 11.5 / 39.9 | 5.2 / 15.8 |
| photo | **adj** | 8 | 3 | 16/32 | 15 (7) | **3.5 / 25.9** | 14.8 / 17.8 | 12.4 / 39.9 | **3.0 / 14.9** |
| Blender example | v03 | 1 | 1 | 4/4 | 4 (0) | 2.9 / 8.9 | — | 2.5 / 14.8 | 0.7 / 3.9 |
| Blender example | **adj** | 1 | 1 | 4/4 | 4 (0) | **0.4 / 6.5** | — | 0.4 / 5.6 | **0.2 / 4.1** |
| **all** | v03 | 41 | 25 | 86/161 | 119 (11) | 2.8 / 8.8 | 4.1 / 17.3 | 3.1 / 9.2 | 1.6 / 4.4 |
| **all** | **adj** | 41 | 25 | 86/161 | 119 (11) | **0.6 / 5.9** | **2.7 / 16.3** | **1.6 / 8.2** | **0.6 / 3.3** |

\* one plan (vertical_5006) read the wrong number as its area ("1189 sq ft" → 17.6 m²) and has no readable
dimensions: the whole plan is 2.5× too small in both configs.

## What the fit changes

- Rooms whose dimensions are printed and read: the drawing now matches them (median 0.6 cm, was 2.8 cm).
- Rooms without printed dimensions also improve (2.7 vs 4.1 cm), because the walls they share move with the fit, and
  the gaps between walls that nothing constrains are kept as the model read them.
- Room areas: 0.6 % median (was 1.6 %).
- Leave-one-out (a room that has printed dimensions, measured as if it had none): 1.6 vs 3.1 cm.

Blender example (truth from the scene): bedrooms 3.14 × 3.34 m drawn as 3.137 × 3.343 (v03: 3.185 × 3.343), areas
10.49 and 10.45 m² for 10.49 (v03: 10.65 and 10.61). The living room is still 6 % short in v03 and adj: the vectorizer
reads the sliding door as a second window line 0.13 m inside the room (topology, not scale).

Villa Bosque Andino (real brochure, references measured by hand in November): primary bedroom 9.35 m² (ref. 9.34;
v03 9.64), bathroom 3.09 (ref. 3.11; v03 3.17), footprint width 7.98 m (ref. 7.82; v03 8.05). Only 2 of its 8 printed
dimensions move walls: the others end at closets, which the fit does not use.

Regression, 8 developer plans with no printed dimensions (scale from the typed area): same rooms and openings,
room areas within ±0.07 m² (the only change there is the uniform wall thickness).

## What the bench says to do next

1. **OCR of the printed dimensions is now the weak link**: 82 of 117 metric dimensions are read, **0 of 40 in feet and
   inches** (the `x` and the `"` are lost or split). 16 of the 40 synthetic plans got no scale at all; 10 of them are
   feet-and-inches plans. A second OCR pass on dimension-like words (digit whitelist, single-line mode) is the next
   cheap win, and the US market depends on it.
2. **Phone photos**: 3 of 8 scaled, 7 of 15 rooms merged, errors of 3–26 cm. The plan has to be rectified (perspective
   and rotation) before anything else.
3. **Topology**: thin partitions next to thick exterior walls still merge rooms; a second line drawn inside a room
   (sliding doors, double windows) eats into it.
4. When two area numbers are read, the first one wins even if it is implausible for the number of rooms.
