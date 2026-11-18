# Furniture bench — results (zmc 0.4.0, 2026-11-18)

Same 4 plans and 92 labels as 0.3.0. Developer images at the sizes the labels were drawn on (Valle Verde at 2400 px).

Before the 0.4 fixture rules, same images: Model marked something there: 72 % · drawn in place: 65 % · right type: **45 %** · precision: 48 % · facing right: 96 %

92 labelled pieces in 4 plans. Model marked something there: 73 % · drawn in place: 67 % · right type: **47 %** · precision: 49 % · facing right: 90 %

| Type | Labels | Model saw | Drawn in place | Right type | Precision | Facing | Width err (m) | Depth err (m) |
|---|---|---|---|---|---|---|---|---|
| chair | 18 | 67 % | 67 % | 50 % | 75 % | 100 % | 0.02 | 0.03 |
| nightstand | 11 | 64 % | 55 % | 45 % | 56 % | 100 % | 0.04 | 0.06 |
| bed | 10 | 100 % | 100 % | 70 % | 78 % | 100 % | 0.08 | 0.08 |
| toilet | 6 | 83 % | 67 % | 33 % | 40 % | 100 % | 0.05 | 0.01 |
| lavatory | 6 | 67 % | 83 % | 17 % | 20 % | 100 % | 0.03 | 0.09 |
| cooktop | 4 | 50 % | 25 % | 25 % | 50 % | 100 % | 0.02 | 0.00 |
| sink | 4 | 50 % | 25 % | 0 % | — | — | — | — |
| fridge | 4 | 50 % | 50 % | 50 % | 67 % | 50 % | 0.04 | 0.04 |
| dining_table | 4 | 100 % | 75 % | 50 % | 100 % | — | 0.15 | 0.12 |
| sofa | 4 | 100 % | 100 % | 50 % | 67 % | 50 % | 0.12 | 0.15 |
| plant | 3 | 67 % | 100 % | 100 % | 60 % | — | 0.04 | 0.01 |
| stool | 3 | 100 % | 100 % | 100 % | 60 % | — | 0.01 | 0.02 |
| coffee_table | 3 | 67 % | 33 % | 33 % | 100 % | — | 0.02 | 0.03 |
| armchair | 3 | 33 % | 33 % | 33 % | 100 % | 100 % | 0.15 | 0.18 |
| washer | 3 | 100 % | 100 % | 33 % | 20 % | 0 % | 0.05 | 0.07 |
| desk | 2 | 50 % | 50 % | 50 % | 12 % | 100 % | 0.19 | 0.02 |
| side_table | 2 | 50 % | 0 % | 0 % | — | — | — | — |
| bathtub | 1 | 100 % | 100 % | 100 % | 100 % | 100 % | 0.29 | 0.05 |
| tv_unit | 1 | 100 % | 100 % | 100 % | 50 % | 100 % | 0.12 | 0.01 |

| Plan | Labels | Model saw | Right type | Drawn | Not matched |
|---|---|---|---|---|---|
| example_brochure | 28 | 26 | 25 | 28 | 2 |
| rivera_del_valle | 25 | 20 | 13 | 23 | 5 |
| valle_verde_A | 21 | 12 | 4 | 21 | 11 |
| valle_verde_C | 18 | 9 | 1 | 16 | 8 |

## What changed in 0.4

- **Bathrooms**: the shower is placed first, as a wall-to-wall rectangle at one end of the bathroom (0.90 m deep), from the image (shower head, drain) or, in a full bathroom with no clue, at the end farthest from the door (assumed). If toilet and lavatory do not fit with it, a 0.90 m corner shower is tried, then none. A white wall-to-wall strip is a bathtub. Toilet and lavatory then go on the same wall, outside the shower and the door swing, with 6 cm between them and 0.45 m free in front; what the model saw well placed is kept. A small round disc read as a second lavatory is the shower head.
- **Kitchens**: the kitchen of an open plan is the longest group of counters that has appliances on or next to it (not the sofa or an L desk); a fridge or washer must be appliance-sized, against a wall and next to those counters, otherwise it is the dining table, a chair or a pouf misread. Cooktop and sink keep 0.30 m apart on the counter.
- **Dining**: chairs at 0.60 m per seat and 0.26 m from the table edge (seat at the edge), long sides full on tables of 1.20 m or more, at most one chair per head, a table lengthened when the brochure shows more chairs than fit, and no chair against a wall or on another piece.

Valle Verde A: its labels sit about 0.2 m off the fixtures in the image, so toilet and lavatory drawn on the real fixtures count as misses here.
