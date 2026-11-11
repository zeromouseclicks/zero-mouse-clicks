"""Command line.

    python -m zmc plan.jpg                       # area read from the image, writes plan_plan-set.pdf
    python -m zmc plan.jpg --area 47 -o out.pdf  # declared private area in m²
    python -m zmc synth -n 4 -o examples/        # synthetic marketing plans (+ ground-truth masks)
"""
import argparse
import os
import sys


def _plan(argv):
    p = argparse.ArgumentParser(prog="zmc", description="Marketing floor-plan image -> architectural plan set (PDF).")
    p.add_argument("image", help="floor-plan image (JPG/PNG)")
    p.add_argument("-o", "--out", help="output PDF (default: <image>_plan-set.pdf)")
    p.add_argument("--area", default="auto",
                   help="declared private / net interior area in m², or 'auto' to read it from the image (default)")
    p.add_argument("--project", default="Marketing floor plan", help="project name for the title block")
    p.add_argument("--unit", default="Unit", help="unit / type name for the title block")
    p.add_argument("--debug", action="store_true", help="also write <out>_debug.jpg (mask and vector model)")
    a = p.parse_args(argv)
    out = a.out or os.path.splitext(a.image)[0] + "_plan-set.pdf"
    area = None if str(a.area).lower() in ("auto", "-", "") else float(a.area)
    from .pipeline import run
    try:
        r = run(a.image, area, a.project, a.unit, out, debug=(os.path.splitext(out)[0] + "_debug.jpg") if a.debug else None)
    except ValueError as e:
        sys.exit(f"zmc: {e}. Pass it with --area <m²>.")
    info = r["info"]
    print(f"Wrote {out}")
    print(f"  footprint {r['W']:.2f} x {r['H']:.2f} m · private area {info.get('area', area)} m²"
          + (" (read from image)" if area is None else ""))
    for x in r["fin"]["rooms"].values():
        print(f"  {x['name']:<28s} {x['area']:6.2f} m²")
    sc = info.get("scale") or {}
    if sc.get("dims") and sc.get("used"):
        print(f"  scale check vs. dimensions printed on the plan: {100 * (sc['used'] / sc['dims'] - 1):+.1f} %")


def _synth(argv):
    p = argparse.ArgumentParser(prog="zmc synth", description="Generate synthetic marketing floor plans with exact labels.")
    p.add_argument("-n", type=int, default=4, help="number of plans")
    p.add_argument("-o", "--out", default="synthetic", help="output folder")
    p.add_argument("--size", type=int, default=768, help="image side in pixels")
    p.add_argument("--style", choices=["co", "us", "line"], help="drawing style (default: random)")
    p.add_argument("--seed", type=int, help="random seed")
    a = p.parse_args(argv)
    import cv2
    import numpy as np
    from . import synth
    if a.seed is not None:
        synth.RNG = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.n):
        im, m, st = synth.render(a.size, a.style)
        cv2.imwrite(os.path.join(a.out, f"synth_{i:03d}_{st}.png"), im)
        cv2.imwrite(os.path.join(a.out, f"synth_{i:03d}_{st}_mask.png"), m)
        cv2.imwrite(os.path.join(a.out, f"synth_{i:03d}_{st}_mask_color.png"), synth.colorize(m))
        print(f"  synth_{i:03d}_{st}.png  net interior area {synth.net_area(m):.1f} m²")
    print(f"Wrote {a.n} plans to {a.out}/  (mask classes: 0 background, 1 space, 2 wall, 3 door, 4 window, "
          f"5 counter/closet, 6 fixture, 7 loose furniture)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "synth":
        _synth(argv[1:])
    else:
        _plan(argv)


if __name__ == "__main__":
    main()
