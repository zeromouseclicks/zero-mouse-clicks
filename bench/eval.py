"""Banco de muebles: compara lo que dibuja Zero Mouse Clicks con muebles marcados a mano.

    python bench/eval.py --images <carpeta con las imágenes de constructoras> [--out bench/results]

Por cada archivo de bench/labels/*.json corre el pipeline completo, pasa las piezas a píxeles de la imagen
original y las empareja con las etiquetas. Mide, por tipo:
  - visto      la etiqueta tiene ALGUNA pieza dibujada en su sitio (el modelo vio algo ahí)
  - acierto    la pieza en su sitio es del tipo correcto
  - precisión  de las piezas dibujadas de ese tipo, cuántas corresponden a una etiqueta de ese tipo
  - orientación de las acertadas con frente conocido, cuántas miran hacia el lado correcto
  - medida     error mediano del ancho y del fondo (m) de las acertadas

Las imágenes de constructoras son solo para examen: nunca se usan para entrenar.
"""
import argparse
import glob
import json
import math
import os
import sys
import tempfile
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from zmc.pipeline import run  # noqa: E402

ORIENTED = {'bed', 'nightstand', 'desk', 'sofa', 'armchair', 'tv_unit', 'toilet', 'lavatory', 'fridge', 'cooktop',
            'sink', 'washer', 'bathtub', 'chair'}
FACE = {0: 'S', 180: 'N', 90: 'E', -90: 'W', -180: 'N', 270: 'W'}


def facing(ang):
    a = ((ang + 180) % 360) - 180
    best = min((0, 90, -90, 180, -180), key=lambda q: abs(a - q))
    return FACE[best]


def to_original(res):
    """Piezas del pipeline -> cajas en píxeles de la imagen original."""
    info, im = res['info'], res['im']
    x0, y0, x1, y1 = info['crop']
    kx, ky = im.shape[1] / float(x1 - x0), im.shape[0] / float(y1 - y0)
    s = res['mod']['m_per_px']
    out = []
    for it in res['fin'].get('items', []):
        a = round(it['ang'] / 90.0) % 2
        hw, hd = it['w'] / s / 2, it['d'] / s / 2
        if abs(it['ang'] % 90) > 1:                       # sillas en diagonal: caja del círculo inscrito
            hw = hd = max(hw, hd) * 0.85
        ex, ey = (hw, hd) if a == 0 else (hd, hw)
        cx, cy = x0 + it['cx'] / kx, y0 + it['cy'] / ky
        out.append(dict(type=it['type'], box=(cx - ex / kx, cy - ey / ky, cx + ex / kx, cy + ey / ky),
                        cx=cx, cy=cy, facing=facing(it['ang']), w=it['w'], d=it['d']))
    raw = []                                               # lo que el modelo marcó como objeto (sin tipo)
    for o in res['fin'].get('objects', []):
        raw.append(dict(box=(x0 + o['x0'] / kx, y0 + o['y0'] / ky, x0 + o['x1'] / kx, y0 + o['y1'] / ky)))
    m_per_orig = s * kx
    return out, raw, m_per_orig


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _hit(gt_box, p_box, tol):
    """La pieza cae en el sitio de la etiqueta: su centro dentro de la caja (agrandada) o mucho traslape."""
    gx0, gy0, gx1, gy1 = gt_box
    cx, cy = _center(p_box)
    pad = tol
    if gx0 - pad <= cx <= gx1 + pad and gy0 - pad <= cy <= gy1 + pad:
        return True
    ix = max(0, min(gx1, p_box[2]) - max(gx0, p_box[0])); iy = max(0, min(gy1, p_box[3]) - max(gy0, p_box[1]))
    return ix * iy >= 0.4 * (gx1 - gx0) * (gy1 - gy0)


def evaluate(label, res):
    preds, raw, mpp = to_original(res)
    tol = 0.12 / mpp                                       # 12 cm de holgura
    gts = label['items']
    used = set()
    rows = []
    # 1) emparejar primero por tipo, luego cualquier pieza en el sitio
    for g in gts:
        gc = _center(g['box'])
        cands = [(i, p) for i, p in enumerate(preds) if i not in used and _hit(g['box'], p['box'], tol)]
        same = [c for c in cands if c[1]['type'] == g['type']]
        pick = min(same or cands, key=lambda c: math.dist(gc, (c[1]['cx'], c[1]['cy'])), default=None)
        r = dict(type=g['type'], seen=pick is not None, correct=bool(pick and pick[1]['type'] == g['type']),
                 raw=any(_hit(g['box'], q['box'], tol) for q in raw))
        if pick:
            used.add(pick[0])
            p = pick[1]
            r['pred'] = p['type']
            if r['correct'] and g.get('facing') and g['type'] in ORIENTED:
                r['face_ok'] = p['facing'] == g['facing']
            if r['correct']:
                gw = (g['box'][2] - g['box'][0]) * mpp; gh = (g['box'][3] - g['box'][1]) * mpp
                gwid, gdep = (gw, gh) if g.get('facing') in ('N', 'S', None) else (gh, gw)
                if g.get('facing') is None:
                    gwid, gdep = sorted((gw, gh))[::-1]
                    pwid, pdep = sorted((p['w'], p['d']))[::-1]
                else:
                    pwid, pdep = p['w'], p['d']
                r['err_w'] = abs(pwid - gwid); r['err_d'] = abs(pdep - gdep)
        rows.append(r)
    extra = [p for i, p in enumerate(preds) if i not in used]
    return rows, preds, extra


def draw(img, label, preds, path):
    im = img.copy()
    t = max(1, im.shape[1] // 500)
    for g in label['items']:
        x0, y0, x1, y1 = [int(v) for v in g['box']]
        cv2.rectangle(im, (x0, y0), (x1, y1), (60, 180, 60), t)
    for p in preds:
        x0, y0, x1, y1 = [int(v) for v in p['box']]
        cv2.rectangle(im, (x0, y0), (x1, y1), (40, 40, 220), t)
        cv2.putText(im, p['type'][:6], (x0, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (40, 40, 220), t)
    s = 1400 / max(im.shape[:2])
    cv2.imwrite(path, cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', help='folder with the developer images (not in the repository)')
    ap.add_argument('--labels', default=os.path.join(os.path.dirname(__file__), 'labels'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allrows, per_plan = [], {}
    n_pred = defaultdict(int); n_pred_ok = defaultdict(int)
    for lf in sorted(glob.glob(os.path.join(a.labels, '*.json'))):
        label = json.load(open(lf))
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        img_path = os.path.join(root, label['path']) if label.get('path') else os.path.join(a.images or '', label['image'])
        if not os.path.exists(img_path):
            print('missing image', img_path); continue
        with tempfile.TemporaryDirectory() as td:
            res = run(img_path, label.get('area'), 'bench', 'bench', os.path.join(td, 'x.pdf'))
        rows, preds, extra = evaluate(label, res)
        for p in preds:
            n_pred[p['type']] += 1
        for r in rows:
            if r['correct']:
                n_pred_ok[r['type']] += 1
        allrows += rows
        name = os.path.splitext(os.path.basename(lf))[0]
        per_plan[name] = dict(labels=len(rows), seen=sum(r['seen'] for r in rows), correct=sum(r['correct'] for r in rows),
                              raw=sum(r['raw'] for r in rows), drawn=len(preds), extra=len(extra))
        draw(cv2.imread(img_path), label, preds, os.path.join(a.out, f'{name}.jpg'))
        print(name, per_plan[name])
    by = defaultdict(list)
    for r in allrows:
        by[r['type']].append(r)
    table = []
    for t in sorted(by, key=lambda t: -len(by[t])):
        rs = by[t]
        fo = [r['face_ok'] for r in rs if 'face_ok' in r]
        ew = [r['err_w'] for r in rs if 'err_w' in r]; ed = [r['err_d'] for r in rs if 'err_d' in r]
        table.append(dict(type=t, n=len(rs), raw=sum(r['raw'] for r in rs), seen=sum(r['seen'] for r in rs),
                          correct=sum(r['correct'] for r in rs), drawn=n_pred.get(t, 0), drawn_ok=n_pred_ok.get(t, 0),
                          face=(sum(fo), len(fo)), err_w=float(np.median(ew)) if ew else None,
                          err_d=float(np.median(ed)) if ed else None))
    tot = dict(n=len(allrows), raw=sum(r['raw'] for r in allrows), seen=sum(r['seen'] for r in allrows),
               correct=sum(r['correct'] for r in allrows), drawn=sum(n_pred.values()),
               face=(sum(r.get('face_ok', 0) for r in allrows), sum(1 for r in allrows if 'face_ok' in r)))
    json.dump(dict(per_plan=per_plan, per_type=table, total=tot), open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    pct = lambda a, b: f'{100 * a / b:.0f} %' if b else '—'
    md = ['# Furniture bench', '',
          f"{tot['n']} labelled pieces in {len(per_plan)} plans. "
          f"Model marked something there: {pct(tot['raw'], tot['n'])} · drawn in place: {pct(tot['seen'], tot['n'])} · "
          f"right type: **{pct(tot['correct'], tot['n'])}** · precision: {pct(sum(n_pred_ok.values()), tot['drawn'])} · "
          f"facing right: {pct(*tot['face'])}", '',
          '| Type | Labels | Model saw | Drawn in place | Right type | Precision | Facing | Width err (m) | Depth err (m) |',
          '|---|---|---|---|---|---|---|---|---|']
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    for r in table:
        md.append(f"| {r['type']} | {r['n']} | {pct(r['raw'], r['n'])} | {pct(r['seen'], r['n'])} | {pct(r['correct'], r['n'])} | "
                  f"{pct(r['drawn_ok'], r['drawn'])} | {pct(*r['face'])} | {fmt(r['err_w'])} | {fmt(r['err_d'])} |")
    md += ['', '| Plan | Labels | Model saw | Right type | Drawn | Not matched |', '|---|---|---|---|---|---|']
    for n, p in per_plan.items():
        md.append(f"| {n} | {p['labels']} | {p['raw']} | {p['correct']} | {p['drawn']} | {p['extra']} |")
    open(os.path.join(a.out, 'results.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
