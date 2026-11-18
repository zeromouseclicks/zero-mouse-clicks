"""Corre el banco de medidas sobre una carpeta de folletos sintéticos (imagen + .json de verdad).

  python bench/measure/run.py <carpeta> <salida.json> [--lean] [--only base,photo] [--configs v03,adj]

Para cada folleto: pipeline hasta los textos (una vez, con caché en <carpeta>/_cache), luego cada
configuración con todas las medidas impresas y dejando fuera la de cada espacio (validación «una fuera»).
"""
import os
import sys
import glob
import json
import pickle
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                      # noqa: E402
from zmc import pipeline as PL   # noqa: E402

from zmc import vectorize5 as V5  # noqa: E402

# v03 = como está publicado (0.3.0); adj = 0.4: grosor uniforme por línea de muro + ajuste local por cotas
CONFIGS = {'v03': dict(adjust=False, uniform=False), 'adj': dict(adjust=True, uniform=True)}


def run_one(path, lean=False, configs=('v03', 'adj')):
    meta = json.load(open(path[:-4] + '.json'))
    cdir = os.path.join(os.path.dirname(path), '_cache'); os.makedirs(cdir, exist_ok=True)
    cp = os.path.join(cdir, os.path.basename(path)[:-4] + ('_lean' if lean else '') + '.pkl')
    t0 = time.time()
    if os.path.exists(cp):
        st = pickle.load(open(cp, 'rb'))
    else:
        st = core.prepare(cv2.imread(path), None, 512 if lean else None)
        pickle.dump(st, open(cp, 'wb'))
    t_prep = time.time() - t0
    byroom = core.dims_by_room(st, meta)
    res = dict(name=os.path.basename(path), variant=meta['variant'], style=meta['style'], unit=meta['unit'],
               n_printed=sum('printed' in g for g in meta['rooms']), t_prep=round(t_prep, 1),
               ocr={g['i']: core.ocr_ok(g, byroom.get(g['i']), meta['unit']) for g in meta['rooms'] if 'printed' in g},
               n_dims_read=len(st['dims']), configs={})
    for cfg in configs:
        PL.ADJUST = CONFIGS[cfg]['adjust']; V5.UNIFORM_LINES = CONFIGS[cfg]['uniform']
        out = dict()
        try:
            mod, im, sc = core.finish(st)
            out['rows'] = core.measure(st, mod, im, meta)
            out['scale_from'] = mod.get('scale_from'); out['adjust'] = mod.get('adjust_info')
            out['dims_fit'] = sc.get('dims_fit')
        except Exception as e:                       # sin escala posible (no leyó medidas ni área)
            out['error'] = f'{type(e).__name__}: {e}'
        loo = {}
        if 'error' not in out:
            for g in meta['rooms']:
                if 'printed' not in g or not byroom.get(g['i']):
                    continue
                keep = [d for i, ds in byroom.items() if i != g['i'] for d in ds]
                try:
                    mod2, im2, _ = core.finish(st, keep)
                    rows = core.measure(st, mod2, im2, meta)
                    loo[g['i']] = [r for r in rows if r['i'] == g['i']][0]
                except Exception as e:
                    loo[g['i']] = dict(error=f'{type(e).__name__}: {e}')
        out['loo'] = loo
        res['configs'][cfg] = out
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    folder, outp = args[0], args[1]
    lean = '--lean' in sys.argv
    only = None; configs = ('v03', 'adj')
    for a in sys.argv:
        if a.startswith('--only='):
            only = a.split('=', 1)[1].split(',')
        if a.startswith('--configs='):
            configs = tuple(a.split('=', 1)[1].split(','))
    files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    if only:
        files = [f for f in files if os.path.basename(f).split('_')[0] in only]
    allres = json.load(open(outp)) if os.path.exists(outp) else {}
    for f in files:
        key = os.path.basename(f) + ('_lean' if lean else '')
        if key in allres and all(c in allres[key]['configs'] for c in configs):
            continue
        r = run_one(f, lean, configs)
        allres[key] = r
        json.dump(allres, open(outp, 'w'), indent=0, default=float)
        c = r['configs'].get(configs[-1], {})
        e = [abs(x) for row in c.get('rows', []) for x in (row.get('ex'), row.get('ey')) if x is not None]
        print(key, 'prep', r['t_prep'], 's', 'err cm', round(100 * float(np.median(e)), 1) if e else '-', c.get('error', ''), flush=True)


if __name__ == '__main__':
    main()
