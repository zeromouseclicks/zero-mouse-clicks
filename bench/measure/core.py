"""Banco de medidas: corre el pipeline hasta la lectura de textos una sola vez (lo caro: modelos y OCR) y
después prueba variantes baratas (con / sin ajuste local, dejando una medida impresa fuera) sobre lo mismo.
Mide las luces libres del modelo final con rayos de cara a cara de muro, igual que se miden en la verdad."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from zmc import pipeline as PL, refine, texts, calib          # noqa: E402
from zmc.vectorize import SP, DR, WN, MS, AP, MU             # noqa: E402
from zmc.vectorize5 import vectorize5                        # noqa: E402


def prepare_steps(im0, area=None, side_max=None):
    """Igual que pipeline.steps hasta antes de vectorizar. Devuelve el estado para `finish`."""
    info = {}
    cands = [(0, float(area), None)] if area is not None else []
    if area is None:
        w1 = yield from texts.read_words_steps(im0, 1600)
        w2 = yield from texts.read_words_steps(im0, 2200)
        cands = texts.find_areas(w1 + w2)
    area0 = cands[0][1] if cands else None
    imc = PL._fit(im0, 448); mc = yield from PL.predict_steps(imc)
    mc, _ = refine.wall_color_filter(imc, mc)
    x0, y0, x1, y1 = refine.plan_bbox(mc)
    k = im0.shape[1] / imc.shape[1]
    crop = im0[int(y0 * k):int(np.ceil(y1 * k)), int(x0 * k):int(np.ceil(x1 * k))]
    info['crop'] = (int(x0 * k), int(y0 * k), int(np.ceil(x1 * k)), int(np.ceil(y1 * k)))
    top = side_max or 640
    side = top
    im = PL._fit(crop, side); m = yield from PL.predict_steps(im, tta=False)
    m1, _ = refine.wall_color_filter(im, m)
    if area0:
        ppm = 1.0 / vectorize5(m1.copy(), area0, image=im)['m_per_px']
        if ppm > 64:
            side = int(np.clip(side * 55 / ppm, min(416, top), top))
            im = PL._fit(crop, side)
    m = yield from PL.predict_steps(im, tta=True)
    m, like = refine.wall_color_filter(im, m, info)
    words = yield from texts.read_words_steps(crop, 2800)
    labs, dims = texts.parse(words)
    kk = im.shape[1] / crop.shape[1]
    for w in words:
        if w['conf'] >= 60 and len(w['text']) >= 3 and w['h'] * kk < 0.08 * max(im.shape[:2]):
            a, b = int(w['x'] * kk), int(w['y'] * kk); c, d = int((w['x'] + w['w']) * kk) + 1, int((w['y'] + w['h']) * kk) + 1
            sub = m[b:d, a:c]; sub[np.isin(sub, (DR, WN, MS, AP, MU))] = SP
    return dict(im=im, m=m, labs=labs, dims=dims, kk=kk, cands=cands, like=like, info=info,
                crop_shape=crop.shape[:2], typed=area is not None)


def prepare(im0, area=None, side_max=None):
    return PL.drive(prepare_steps(im0, area, side_max))


def finish(st, dims=None):
    """Vectoriza y escala (pipeline._scaled_model) con las medidas `dims` (None = todas las leídas)."""
    dims = st['dims'] if dims is None else dims
    mod, im, m, labs, dims2, sc = PL._scaled_model(st['im'].copy(), st['m'].copy(), st['labs'], dims, st['kk'],
                                                    st['cands'], st['like'], typed=st['typed'])
    mod['stretch'] = sc.get('stretch_corrected')
    return mod, im, sc


def to_model(st, mod, im, x, y):
    """Punto de la imagen original -> px del modelo final (recorte, escala, estiramiento y ajuste local)."""
    cx0, cy0 = st['info']['crop'][:2]
    ch, cw = st['crop_shape']
    h0, w0 = st['im'].shape[:2]
    xm = (x - cx0) * w0 / cw
    ym = (y - cy0) * h0 / ch
    f = (mod.get('stretch') or 1.0)
    ym *= f
    wp = mod.get('warp')
    if wp:
        xm = float(np.interp(xm, wp['x_old'], wp['x_new']))
        ym = float(np.interp(ym, wp['y_old'], wp['y_new']))
    return xm, ym


def _runs(A, lab, label, axis):
    """Luces (px) del espacio `label`: tramos libres en filas/columnas al 30/50/70 % de su caja."""
    ys, xs = np.nonzero(lab == label)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = []
    for f in (0.3, 0.5, 0.7):
        if axis == 'h':
            y = int(y0 + f * (y1 - y0)); row = A[y]; inside = lab[y] == label
        else:
            x = int(x0 + f * (x1 - x0)); row = A[:, x]; inside = lab[:, x] == label
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            continue
        c = int(np.median(idx))
        if row[c]:
            continue
        l = c
        while l > 0 and not row[l - 1]:
            l -= 1
        r = c
        while r < len(row) - 1 and not row[r + 1]:
            r += 1
        out.append(r + 1 - l)
    return float(np.median(out)) if out else None


def measure(st, mod, im, meta):
    """Compara cada espacio de la verdad con el espacio del modelo que contiene su centro."""
    s = mod['m_per_px']
    A, _ = calib.obstacles(mod, None)
    lab = mod['room_labels']
    by_label = {r['label']: r for r in mod['rooms']}
    net = mod.get('net_px') or calib.regrow(mod)
    rows = []
    seen = {}
    for g in meta['rooms']:
        x, y = to_model(st, mod, im, g['cx_img'], g['cy_img'])
        li = texts._room_at(lab, x, y, int(2.5 * mod['thickness_px']))
        r = by_label.get(li)
        row = dict(i=g['i'], name=g['name'], type=g['type'], printed='printed' in g, sx=g['sx'], sy=g['sy'], area=g['area'])
        if r is None:
            row['found'] = False
            rows.append(row); continue
        row['found'] = True
        seen.setdefault(li, []).append(g['i'])
        sx = _runs(A, lab, li, 'h'); sy = _runs(A, lab, li, 'v')
        row['mx'] = sx * s if sx else None
        row['my'] = sy * s if sy else None
        row['marea'] = net.get(r['id'], 0) * s * s
        rows.append(row)
    for row in rows:                       # dos espacios de la verdad en uno del modelo: topología, no medida
        row['merged'] = False
    for li, ids in seen.items():
        if len(ids) > 1:
            for row in rows:
                if row['i'] in ids:
                    row['merged'] = True
    for row in rows:
        if row.get('found') and not row['merged'] and row.get('mx') and row.get('my'):
            row['ex'] = row['mx'] - row['sx']; row['ey'] = row['my'] - row['sy']
            row['earea'] = row['marea'] / row['area'] - 1
    return rows


def dims_by_room(st, meta):
    """Medidas leídas por el OCR -> espacio de la verdad más cercano (en coordenadas del recorte)."""
    cx0, cy0 = st['info']['crop'][:2]
    out = {}
    for d in st['dims']:
        best, bi = None, None
        for g in meta['rooms']:
            dd = np.hypot(g['cx_img'] - cx0 - d['x'], g['cy_img'] - cy0 - d['y'])
            if best is None or dd < best:
                best, bi = dd, g['i']
        out.setdefault(bi, []).append(d)
    return out


def ocr_ok(g, ds, unit):
    """¿El OCR leyó bien la medida impresa de este espacio? (mismos dos valores, en cualquier orden)."""
    if 'printed' not in g:
        return None
    tol = 0.006 if unit == 'm' else 0.013
    p = sorted(g['printed'])
    for d in ds or []:
        q = sorted([d['a'], d['b']])
        if abs(q[0] - p[0]) <= tol and abs(q[1] - p[1]) <= tol:
            return True
    return False
