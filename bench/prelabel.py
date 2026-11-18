"""Pre-etiquetas: corre el pipeline y guarda sus piezas como un JSON de etiquetas para corregir en label.html.

    python bench/prelabel.py plano.jpg [--area 51] [-o bench/labels/plano.json]

Corregir es mucho más rápido que marcar desde cero; revise cada pieza antes de usarla como verdad.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import to_original, ORIENTED  # noqa: E402
from zmc.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--area', type=float)
    ap.add_argument('-o', '--out')
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        res = run(a.image, a.area, 'bench', 'bench', os.path.join(td, 'x.pdf'))
    preds, _, _ = to_original(res)
    items = [dict(type=p['type'], box=[round(v) for v in p['box']], facing=p['facing'] if p['type'] in ORIENTED else None)
             for p in preds if p['type'] != 'misc']
    out = a.out or os.path.splitext(a.image)[0] + '.json'
    d = dict(image=os.path.basename(a.image), note='PRE-LABELS from the pipeline: correct every item before using.', items=items)
    if a.area:
        d['area'] = a.area
    json.dump(d, open(out, 'w'), indent=1)
    print(f'{len(items)} pieces -> {out}')


if __name__ == '__main__':
    main()
