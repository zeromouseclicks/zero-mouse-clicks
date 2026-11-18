"""Muebles y aparatos como piezas de catálogo (v0.3), solo con reglas.

El modelo de 8 clases entrega manchas de «mueble suelto», «aparato» y «mesón». Aquí cada mancha se vuelve
una pieza con tipo (cama, mesa de noche, sofá, comedor, silla...), medida de catálogo y orientación
(hacia dónde mira su frente), para dibujarla después con un bloque paramétrico (blocks.py).

Salida: lista de piezas en píxeles de la imagen del modelo:
    dict(type, cx, cy, w, d, ang, room, src)
w = ancho (m, a lo largo del muro de respaldo), d = fondo (m). ang en grados, convención de la lámina
(Y hacia arriba): 0 = el frente mira hacia abajo de la lámina, 90 = a la derecha, 180 = arriba, -90 = izquierda.
"""
import math

import cv2
import numpy as np

from .vectorize import MS, AP, MU

FURNITURE = ('bed', 'nightstand', 'desk', 'chair', 'armchair', 'sofa', 'coffee_table', 'side_table', 'tv_unit',
             'dining_table', 'stool', 'plant', 'misc')
FIXTURES = ('fridge', 'cooktop', 'sink', 'washer', 'toilet', 'lavatory', 'shower', 'bathtub')
NAMES = {'bed': 'Bed', 'nightstand': 'Nightstand', 'desk': 'Desk', 'chair': 'Chair', 'armchair': 'Armchair',
         'sofa': 'Sofa', 'coffee_table': 'Coffee table', 'side_table': 'Side table', 'tv_unit': 'TV unit',
         'dining_table': 'Dining table', 'stool': 'Bar stool', 'plant': 'Plant', 'misc': 'Other furniture',
         'fridge': 'Refrigerator', 'cooktop': 'Cooktop', 'sink': 'Kitchen sink', 'washer': 'Washer / dryer',
         'toilet': 'Toilet', 'lavatory': 'Lavatory', 'shower': 'Shower', 'bathtub': 'Bathtub'}

BED_W = (0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00)       # anchos de catálogo (Colombia / EE. UU. aprox.)
ANG = {'N': 0.0, 'S': 180.0, 'W': 90.0, 'E': -90.0}      # lado del respaldo en la imagen -> ángulo del frente
BACK = {v: k for k, v in ANG.items()}
OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


def _snap(v, opts):
    return min(opts, key=lambda o: abs(o - v))


def _dims(o):
    return sorted((o['w_m'], o['h_m']))


class _Ctx:
    def __init__(self, image, mask, model):
        self.im, self.mask, self.mod = image, mask, model
        self.s = model['m_per_px']
        self.lab = model['room_labels']
        self.H, self.W = self.lab.shape

    def px(self, m):
        return m / self.s

    def side_dist(self, rl, box, side, max_m=1.2):
        """Distancia (m) desde un lado de la caja hasta el borde del espacio (muro), mediana de 5 rayos."""
        x0, y0, x1, y1 = box
        maxpx = int(self.px(max_m))
        if side in 'NS':
            span = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 1 - 0.2 * (x1 - x0), 5)
        else:
            span = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 1 - 0.2 * (y1 - y0), 5)
        out = []
        for t in span:
            t = int(round(t)); d = 0
            for step in range(maxpx):
                if side == 'N':
                    x, y = t, y0 - 1 - step
                elif side == 'S':
                    x, y = t, y1 + step
                elif side == 'W':
                    x, y = x0 - 1 - step, t
                else:
                    x, y = x1 + step, t
                if not (0 <= x < self.W and 0 <= y < self.H) or self.lab[y, x] != rl:
                    break
                d += 1
            out.append(d)
        return float(np.median(out)) * self.s

    def dists(self, rl, box):
        return {sd: self.side_dist(rl, box, sd) for sd in 'NSEW'}


def _box(o):
    return (o['x0'], o['y0'], o['x1'], o['y1'])


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _merge(cands, s):
    """Fragmentos de una misma pieza: lo contenido en otra caja (≥ 60 %) se absorbe."""
    cands = sorted(cands, key=lambda b: -_area(b['box']))
    out = []
    for c in cands:
        small = _area(c['box']) * s * s < 0.06
        host = next((o for o in out if _inter(o['box'], c['box']) >= 0.6 * _area(c['box'])
                     and (small or o.get('fill', 0.7) >= 0.6)), None)
        if host is None:
            out.append(dict(c))
    for o in out:
        x0, y0, x1, y1 = o['box']
        o['w_m'], o['h_m'] = (x1 - x0) * s, (y1 - y0) * s
    return out


def _core(ctx, rl, box, k_m):
    """Núcleo de una mancha grande: apertura con un elemento de k_m metros (separa sillas de la mesa,
    escritorio de la cama). Devuelve las cajas de los componentes que sobreviven, de mayor a menor."""
    x0, y0, x1, y1 = box
    sub = np.isin(ctx.mask[y0:y1, x0:x1], (MU, AP, MS)) & (ctx.lab[y0:y1, x0:x1] == rl)
    sub = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k = max(3, int(ctx.px(k_m)) | 1)
    core = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, l, st, _ = cv2.connectedComponentsWithStats(core, 8)
    boxes = [(x0 + st[i][0], y0 + st[i][1], x0 + st[i][0] + st[i][2], y0 + st[i][1] + st[i][3], st[i][4])
             for i in range(1, n)]
    return sorted(boxes, key=lambda b: -b[4]), sub


def _orient_box(box, back, ctx):
    """(ancho, fondo) en metros de una caja según el lado del respaldo."""
    w = (box[2] - box[0]) * ctx.s
    h = (box[3] - box[1]) * ctx.s
    return (w, h) if back in 'NS' else (h, w)


def _place(ctx, rl, box, back, w, d, typ, room, flush=0.45, src='model', dist=None):
    """Pieza con respaldo en `back`: centro lateral de la caja; respaldo pegado al muro si está cerca."""
    x0, y0, x1, y1 = box
    dist = dist if dist is not None else ctx.side_dist(rl, box, back)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dp = ctx.px(d)
    if back == 'N':
        face = y0 - (ctx.px(dist) if dist <= flush else 0)
        cy = face + dp / 2
    elif back == 'S':
        face = y1 + (ctx.px(dist) if dist <= flush else 0)
        cy = face - dp / 2
    elif back == 'W':
        face = x0 - (ctx.px(dist) if dist <= flush else 0)
        cx = face + dp / 2
    else:
        face = x1 + (ctx.px(dist) if dist <= flush else 0)
        cx = face - dp / 2
    return dict(type=typ, cx=float(cx), cy=float(cy), w=float(w), d=float(d), ang=ANG[back], room=room, src=src)


def _free(typ, box, w, d, ang, room, src='model'):
    return dict(type=typ, cx=(box[0] + box[2]) / 2.0, cy=(box[1] + box[3]) / 2.0, w=float(w), d=float(d),
                ang=float(ang), room=room, src=src)


def _ang_to(ctx, cx, cy, tx, ty):
    """Ángulo (convención de lámina) para que el frente mire hacia (tx, ty) en píxeles."""
    ux, uy = tx - cx, -(ty - cy)
    a = math.degrees(math.atan2(ux, -uy))
    return a


def _snap_ang(a, step=90.0):
    return round(a / step) * step


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _gap(a, b, s):
    """Distancia (m) entre dos cajas (0 si se tocan)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy) * s


# ---------------------------------------------------------------------------------------------- alcobas
def _assemble_beds(ctx, cands):
    """Cobija y almohadas leídas por separado: piezas alineadas en un eje, con poco espacio entre ellas,
    que juntas tienen medida de cama, se unen."""
    s = ctx.s
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                A, B = a['box'], b['box']
                for ax in (0, 1):
                    lo_a, hi_a = (A[0], A[2]) if ax == 0 else (A[1], A[3])
                    lo_b, hi_b = (B[0], B[2]) if ax == 0 else (B[1], B[3])
                    ov = min(hi_a, hi_b) - max(lo_a, lo_b)
                    if ov < 0.6 * min(hi_a - lo_a, hi_b - lo_b):
                        continue
                    if abs((hi_a - lo_a) - (hi_b - lo_b)) * s > 0.35:
                        continue
                    U = (min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]))
                    gap = ((max(A[1], B[1]) - min(A[3], B[3])) if ax == 0 else (max(A[0], B[0]) - min(A[2], B[2]))) * s
                    lo, hi = sorted(((U[2] - U[0]) * s, (U[3] - U[1]) * s))
                    if gap <= 0.7 and 0.8 <= lo <= 2.1 and 1.6 <= hi <= 2.3 and max(_dims(a)[1], _dims(b)[1]) < 1.6:
                        a.update(box=U, w_m=(U[2] - U[0]) * s, h_m=(U[3] - U[1]) * s, fill=0.8)
                        cands.remove(b)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break


def _bedrooms(ctx, r, cands, items):
    rl, rid, s = r['label'], r['id'], ctx.s
    _assemble_beds(ctx, cands)
    used = set()
    beds = []
    big = [c for c in cands if _dims(c)[0] >= 0.75 and _dims(c)[1] >= 1.3 and c['w_m'] * c['h_m'] >= 1.1]
    for c in sorted(big, key=lambda c: -c['w_m'] * c['h_m'])[:2]:
        box = c['box']
        if c['fill'] < 0.62 and c['w_m'] * c['h_m'] > 2.2:     # cama pegada a escritorio, silla o mesa de noche
            cores, _ = _core(ctx, rl, box, 0.55)
            if cores and (cores[0][2] - cores[0][0]) * s >= 0.75 and (cores[0][3] - cores[0][1]) * s >= 1.3:
                box = cores[0][:4]
                for q in cores[1:]:
                    if q[4] * s * s >= 0.08:
                        cands.append(dict(box=q[:4], w_m=(q[2] - q[0]) * s, h_m=(q[3] - q[1]) * s, fill=0.8, derived=True))
        if beds and _gap(beds[0]['box'], box, s) < 0.05:
            continue
        beds.append(dict(box=box, src=c))
        used.add(id(c))
    for b in beds:
        box = b['box']
        dd = ctx.dists(rl, box)
        wm, hm = (box[2] - box[0]) * s, (box[3] - box[1]) * s
        # la cabecera va en un lado corto (o en cualquiera si la cama es casi cuadrada) y contra el muro más cercano
        sides = [sd for sd in 'NSEW' if (sd in 'NS' and wm <= hm + 0.25) or (sd in 'EW' and hm <= wm + 0.25)] or list('NSEW')
        back = min(sides, key=lambda sd: dd[sd])
        if len(sides) > 1:
            a, bb = sorted(sides, key=lambda sd: dd[sd])[:2]
            if abs(dd[a] - dd[bb]) < 0.15:          # empate: las almohadas (banda clara) marcan la cabecera
                back = max((a, bb), key=lambda sd: _bright(ctx, box, sd))
        # mesas de noche a lado y lado de la cabecera; si el modelo las pegó a la cama, la cama se recorta
        x0, y0, x1, y1 = box
        ns = []
        for c in cands:
            if id(c) in used or c.get('derived_bed'):
                continue
            lo, hi = _dims(c)
            if not (0.22 <= lo and hi <= 0.8):
                continue
            cx, cy = _center(c['box'])
            if back in 'NS':
                head = y0 if back == 'N' else y1
                near_head = abs(cy - head) <= ctx.px(0.55) or (head == y0 and cy < y0 + ctx.px(0.6)) or (head == y1 and cy > y1 - ctx.px(0.6))
                left, right = cx < (x0 + x1) / 2, cx >= (x0 + x1) / 2
                lat_ok = (cx <= x0 + ctx.px(0.4) and cx >= x0 - ctx.px(0.8)) or (cx >= x1 - ctx.px(0.4) and cx <= x1 + ctx.px(0.8))
            else:
                head = x0 if back == 'W' else x1
                near_head = abs(cx - head) <= ctx.px(0.55) or (head == x0 and cx < x0 + ctx.px(0.6)) or (head == x1 and cx > x1 - ctx.px(0.6))
                left, right = cy < (y0 + y1) / 2, cy >= (y0 + y1) / 2
                lat_ok = (cy <= y0 + ctx.px(0.4) and cy >= y0 - ctx.px(0.8)) or (cy >= y1 - ctx.px(0.4) and cy <= y1 + ctx.px(0.8))
            if near_head and lat_ok:
                ns.append((c, left))
        nb = list(box)
        for c, left in ns:
            q = c['box']
            if back in 'NS':
                if left and q[2] > nb[0]:
                    nb[0] = min(q[2], nb[0] + ctx.px(0.6))
                elif not left and q[0] < nb[2]:
                    nb[2] = max(q[0], nb[2] - ctx.px(0.6))
            else:
                if left and q[3] > nb[1]:
                    nb[1] = min(q[3], nb[1] + ctx.px(0.6))
                elif not left and q[1] < nb[3]:
                    nb[3] = max(q[1], nb[3] - ctx.px(0.6))
        box = tuple(nb)
        width, length = _orient_box(box, back, ctx)
        width = _snap(min(width, 2.05), BED_W)
        length = 1.90 if width <= 1.0 else 2.00
        it = _place(ctx, rl, box, back, width, length, 'bed', rid, flush=0.6, dist=dd[back])
        items.append(it)
        b['item'] = it
        ns_kept = []
        for c, left in ns:
            if any(_gap(c['box'], q['box'], s) < 0.05 and left == l2 for q, l2 in ns_kept):
                used.add(id(c)); continue
            ns_kept.append((c, left))
        for c, left in ns_kept[:2]:
            items.append(_place(ctx, rl, c['box'], back, 0.45, 0.40, 'nightstand', rid, flush=0.45))
            used.add(id(c))
    # escritorio contra un muro y su silla
    rest = [c for c in cands if id(c) not in used]
    for c in rest:
        lo, hi = _dims(c)
        if 0.35 <= lo <= 0.85 and 0.8 <= hi <= 2.0 and hi / lo >= 1.5:
            dd = ctx.dists(rl, c['box'])
            long_sides = 'NS' if c['w_m'] >= c['h_m'] else 'EW'
            back = min(long_sides, key=lambda sd: dd[sd])
            if dd[back] <= 0.35:
                w, d = _orient_box(c['box'], back, ctx)
                it = _place(ctx, rl, c['box'], back, round(min(max(w, 0.8), 1.8) / 0.05) * 0.05,
                            0.60 if d >= 0.55 else 0.50, 'desk', rid, flush=0.35, dist=dd[back])
                items.append(it); used.add(id(c))
                for q in rest:
                    if id(q) in used:
                        continue
                    ql, qh = _dims(q)
                    if 0.3 <= ql and qh <= 0.75 and _gap(q['box'], c['box'], s) <= 0.45:
                        qc = _center(q['box'])
                        items.append(_free('chair', q['box'], 0.48, 0.48,
                                           _snap_ang(_ang_to(ctx, *qc, it['cx'], it['cy'])), rid))
                        used.add(id(q))
                        break
    return used


def _bright(ctx, box, side):
    """Brillo medio de la franja del 22 % junto a un lado (almohadas blancas en la cabecera)."""
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(ctx.im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if g.size == 0:
        return 0.0
    h, w = g.shape
    t = max(1, int(0.22 * (h if side in 'NS' else w)))
    band = {'N': g[:t], 'S': g[h - t:], 'W': g[:, :t], 'E': g[:, w - t:]}[side]
    return float(band.mean())


def _roundish(ctx, box):
    """Mesa redonda u ovalada: las esquinas de su caja no tienen el color del centro."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return False
    lab = cv2.cvtColor(ctx.im[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    c = lab[int(0.35 * h):int(0.65 * h) + 1, int(0.35 * w):int(0.65 * w) + 1].reshape(-1, 3)
    cm = np.median(c, 0)
    k = max(2, int(0.13 * min(w, h)))
    corners = [lab[:k, :k], lab[:k, w - k:], lab[h - k:, :k], lab[h - k:, w - k:]]
    diff = sum(1 for q in corners if np.linalg.norm(np.median(q.reshape(-1, 3), 0) - cm) > 14)
    mx, my, kk = w // 2, h // 2, max(1, k // 2)
    mids = [lab[:k, mx - kk:mx + kk + 1], lab[h - k:, mx - kk:mx + kk + 1], lab[my - kk:my + kk + 1, :k], lab[my - kk:my + kk + 1, w - k:]]
    same = sum(1 for q in mids if np.linalg.norm(np.median(q.reshape(-1, 3), 0) - cm) <= 14)
    return diff >= 3 or (diff >= 2 and same >= 3)


# ---------------------------------------------------------------------------------------------- sala / comedor
def _living(ctx, r, cands, items, islands):
    rl, rid, s = r['label'], r['id'], ctx.s
    used = set()
    anc = r.get('kitchen_anchor')
    # banquetas a lo largo de la barra / isla
    for isl in islands:
        ib = _box(isl)
        horiz = (ib[2] - ib[0]) >= (ib[3] - ib[1])
        for c in cands:
            if id(c) in used:
                continue
            lo, hi = _dims(c)
            if not (0.2 <= lo <= 0.65 and hi <= 1.6):
                continue
            cx, cy = _center(c['box'])
            if horiz:
                along = ib[0] - ctx.px(0.1) <= cx <= ib[2] + ctx.px(0.1)
                off = min(abs(c['box'][1] - ib[3]), abs(ib[1] - c['box'][3])) * s
                inside = ib[1] <= cy <= ib[3]
            else:
                along = ib[1] - ctx.px(0.1) <= cy <= ib[3] + ctx.px(0.1)
                off = min(abs(c['box'][0] - ib[2]), abs(ib[0] - c['box'][2])) * s
                inside = ib[0] <= cx <= ib[2]
            if along and (off <= 0.55 or inside):
                n = max(1, int(round(hi / 0.45))) if hi > 0.6 else 1
                for k in range(n):
                    if horiz:
                        bx0 = c['box'][0] + (c['box'][2] - c['box'][0]) * k / n
                        bx1 = c['box'][0] + (c['box'][2] - c['box'][0]) * (k + 1) / n
                        sb = (bx0, c['box'][1], bx1, c['box'][3])
                    else:
                        by0 = c['box'][1] + (c['box'][3] - c['box'][1]) * k / n
                        by1 = c['box'][1] + (c['box'][3] - c['box'][1]) * (k + 1) / n
                        sb = (c['box'][0], by0, c['box'][2], by1)
                    sc = _center(sb)
                    items.append(_free('stool', sb, 0.40, 0.40, _snap_ang(_ang_to(ctx, *sc, *_center(ib))), rid))
                used.add(id(c))
    # sofá: largo, de fondo 0,6–1,15 m; respaldo hacia el muro más cercano o de espaldas al centro de la sala
    sofas, sofa_boxes = [], []
    for c in sorted(cands, key=lambda c: -max(c['w_m'], c['h_m'])):
        if id(c) in used:
            continue
        lo, hi = _dims(c)
        if 0.55 <= lo <= 1.2 and 1.3 <= hi <= 3.4 and hi / lo >= 1.6:
            if anc is not None and math.dist(_center(c['box']), anc) * s < 1.2:
                continue
            dd = ctx.dists(rl, c['box'])
            long_sides = 'NS' if c['w_m'] >= c['h_m'] else 'EW'
            back = min(long_sides, key=lambda sd: dd[sd])
            if dd[back] > 0.8:                       # sofá exento: de espaldas al centro del espacio
                rx, ry, rw, rh = r['bbox']
                cx, cy = _center(c['box'])
                if long_sides == 'NS':
                    back = 'N' if cy < ry + rh / 2 else 'S'
                else:
                    back = 'W' if cx < rx + rw / 2 else 'E'
            w, _ = _orient_box(c['box'], back, ctx)
            it = _place(ctx, rl, c['box'], back, round(min(max(w, 1.4), 3.2) / 0.1) * 0.1, 0.90, 'sofa', rid,
                        flush=0.5, dist=dd[back])
            items.append(it); sofas.append(it); sofa_boxes.append(c['box']); used.add(id(c))
            if len(sofas) >= 2:
                break
    # mueble de TV: bajo, largo y contra un muro
    for c in cands:
        if id(c) in used:
            continue
        if any(_gap(c['box'], sb, s) < 0.12 for sb in sofa_boxes):
            used.add(id(c)); continue      # cola del sofá (brazo, cojín) leída aparte
        lo, hi = _dims(c)
        if 0.2 <= lo <= 0.6 and 0.9 <= hi <= 2.6 and hi / lo >= 2.5:
            dd = ctx.dists(rl, c['box'])
            back = min('NS' if c['w_m'] >= c['h_m'] else 'EW', key=lambda sd: dd[sd])
            if dd[back] <= 0.3:
                w, _ = _orient_box(c['box'], back, ctx)
                items.append(_place(ctx, rl, c['box'], back, round(min(max(w, 1.0), 2.4) / 0.1) * 0.1, 0.40,
                                    'tv_unit', rid, flush=0.3, dist=dd[back]))
                used.add(id(c))
    # mesa de centro: la pieza más cercana al frente del sofá, lejos de los muros
    coffee = []
    best = None
    for c in cands:
        if id(c) in used or not sofas:
            continue
        lo, hi = _dims(c)
        if 0.35 <= lo and hi <= 1.5 and hi / max(lo, 1e-3) <= 2.2:
            cx, cy = _center(c['box'])
            sf = min(sofas, key=lambda it: math.dist((cx, cy), (it['cx'], it['cy'])))
            dsf = math.dist((cx, cy), (sf['cx'], sf['cy'])) * s
            dd = ctx.dists(rl, c['box'])
            if dsf <= 1.9 and min(dd.values()) >= 0.25 and _in_front(sf, cx, cy, ctx):
                if best is None or dsf < best[0]:
                    best = (dsf, c)
    if best:
        c = best[1]
        lo, hi = _dims(c)
        rnd = hi / lo < 1.3 and (c['fill'] < 0.8 or _roundish(ctx, c['box']))
        w, d = min(max(c['w_m'], 0.45), 1.3), min(max(c['h_m'], 0.45), 1.3)
        if rnd:
            w = d = round((w + d) / 2 / 0.05) * 0.05
        it = _free('coffee_table', c['box'], w, d, 0, rid)
        it['round'] = bool(rnd)
        items.append(it); coffee.append(it); used.add(id(c))
    # poltronas: cuadradas, cerca de la mesa de centro o del sofá, mirando hacia ella
    focus = coffee[0] if coffee else (sofas[0] if sofas else None)
    for c in cands:
        if id(c) in used or focus is None:
            continue
        lo, hi = _dims(c)
        if 0.5 <= lo and hi <= 1.15 and hi / lo <= 1.45:
            cx, cy = _center(c['box'])
            if math.dist((cx, cy), (focus['cx'], focus['cy'])) * s <= 2.4:
                a = _snap_ang(_ang_to(ctx, cx, cy, focus['cx'], focus['cy']))
                sz = round(min(max((lo + hi) / 2, 0.7), 0.95) / 0.05) * 0.05
                items.append(_free('armchair', c['box'], sz, sz, a, rid)); used.add(id(c))
    # comedor: mesa exenta; sus sillas son las piezas pequeñas alrededor (o las de un núcleo partido)
    def chair_score(c):
        near = sum(1 for q in cands if q is not c and id(q) not in used and 0.18 <= _dims(q)[0] and _dims(q)[1] <= 0.75
                   and _gap(q['box'], c['box'], s) <= 0.5)
        return near + (1 if c['fill'] >= 0.75 else 0) - (2 if c.get('was_counter') and not near else 0)
    for c in sorted(cands, key=lambda c: (-chair_score(c), -c['w_m'] * c['h_m'])):
        if id(c) in used:
            continue
        lo, hi = _dims(c)
        if not (0.6 <= lo <= 1.6 and 0.7 <= hi <= 2.8):
            continue
        dd = ctx.dists(rl, c['box'])
        if min(dd.values()) < 0.15 and hi < 1.2:
            continue
        cxy = _center(c['box'])
        if any(math.dist(cxy, (q['cx'], q['cy'])) * s < 1.3 for q in coffee) or \
                any(_in_front(q, *cxy, ctx) and math.dist(cxy, (q['cx'], q['cy'])) * s < 1.6 for q in sofas):
            continue
        box = c['box']
        extra = []
        core_fill = c['fill']
        cores, sub = _core(ctx, rl, box, 0.5)
        if cores and (cores[0][2] - cores[0][0]) * s >= 0.55 and (cores[0][3] - cores[0][1]) * s >= 0.55:
            tb = cores[0][:4]
            core_fill = cores[0][4] / float(max(1, _area(tb)))
            # sillas pegadas: lo que queda de la mancha fuera de la mesa
            x0, y0 = box[0], box[1]
            rest = sub.copy()
            kk = max(1, int(ctx.px(0.06)))
            rest[max(0, tb[1] - y0 - kk):tb[3] - y0 + kk, max(0, tb[0] - x0 - kk):tb[2] - x0 + kk] = 0
            n2, l2, st2, _ = cv2.connectedComponentsWithStats(rest, 8)
            extra = [(x0 + st2[i][0], y0 + st2[i][1], x0 + st2[i][0] + st2[i][2], y0 + st2[i][1] + st2[i][3])
                     for i in range(1, n2) if st2[i][4] * s * s >= 0.04]
            box = tb
        tw, th = (box[2] - box[0]) * s, (box[3] - box[1]) * s
        lo2, hi2 = sorted((tw, th))
        rnd = hi2 / lo2 < 1.3 and lo2 <= 1.3 and (core_fill < 0.85 or _roundish(ctx, box))
        if rnd:
            dia = _snap((lo2 + hi2) / 2, (0.80, 0.90, 1.00, 1.10, 1.20))
            tw = th = dia
        else:
            a, b = _snap(lo2, (0.75, 0.80, 0.90, 1.00)), _snap(hi2, (0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.40))
            tw, th = (b, a) if tw >= th else (a, b)
        t = _free('dining_table', box, tw, th, 0, rid)
        t['round'] = bool(rnd)
        items.append(t); used.add(id(c))
        # sillas: piezas pequeñas cerca del borde de la mesa
        chairs = [cb for cb in extra]
        for q in cands:
            if id(q) in used:
                continue
            ql, qh = _dims(q)
            if 0.18 <= ql and qh <= 0.75 and _gap(q['box'], box, s) <= 0.55:
                chairs.append(q['box']); used.add(id(q))
        _chairs(ctx, t, chairs, items, rid, rl)
        break
    return used


def _in_front(it, x, y, ctx):
    """¿(x, y) está del lado del frente de la pieza?"""
    a = math.radians(it['ang'])
    fx, fy = math.sin(a), -math.cos(a)            # frente en coordenadas de lámina (Y arriba)
    vx, vy = (x - it['cx']), -(y - it['cy'])
    return fx * vx + fy * vy > 0


CHAIR_W, CHAIR_D = 0.45, 0.47     # silla de comedor de catálogo
CHAIR_OFF = 0.26                  # centro de la silla a 0,26 m del borde: el asiento queda justo al borde de la mesa
CHAIR_PITCH = 0.60                # un puesto de comedor ocupa ~0,60 m de borde


def _chairs(ctx, t, boxes, items, rid, rl=None):
    """Sillas alrededor de la mesa (v0.4). Del folleto se toma cuántas sillas hay en cada lado; se ponen a la
    distancia y el paso de un comedor real (0,60 m por puesto, el asiento al borde de la mesa), parejas en los lados
    largos, a lo sumo una en cada cabecera, y nunca contra un muro u otro mueble."""
    s = ctx.s
    cx, cy = t['cx'], t['cy']
    hw, hh = ctx.px(t['w']) / 2, ctx.px(t['d']) / 2
    off = ctx.px(CHAIR_OFF)
    read, widths = [], []
    for b in boxes:
        bx, by = _center(b)
        if math.dist((bx, by), (cx, cy)) * s >= 0.25:
            read.append((bx, by)); widths.append(((b[2] - b[0]) * s, (b[3] - b[1]) * s))
    seats = []
    if t.get('round'):
        dia = t['w']
        cap = 4 if dia < 1.05 else 5 if dia < 1.2 else 6
        n = len(read)
        n = 4 if n == 0 else min(cap, max(2, n + (n % 2 if n < cap else 0)))
        step = 2 * math.pi / n
        if read:                                      # fase de las sillas leídas (diagonales o ejes)
            z = sum(complex(math.cos(n * math.atan2(by - cy, bx - cx)), math.sin(n * math.atan2(by - cy, bx - cx)))
                    for bx, by in read)
            phase = math.atan2(z.imag, z.real) / n
        else:
            phase = 0.0
        R = hw + off
        for k in range(n):
            a = phase + k * step
            seats.append((cx + R * math.cos(a), cy + R * math.sin(a)))
    else:
        horiz = t['w'] >= t['d']
        L, Sd = (t['w'], t['d']) if horiz else (t['d'], t['w'])
        cnt = {'l+': 0, 'l-': 0, 's+': 0, 's-': 0}
        for (bx, by), (bw, bh) in zip(read, widths):
            u, v = (bx - cx) / max(hw, 1), (by - cy) / max(hh, 1)
            if horiz:
                side = ('l+' if v > 0 else 'l-') if abs(v) >= abs(u) else ('s+' if u > 0 else 's-')
            else:
                side = ('l+' if u > 0 else 'l-') if abs(u) >= abs(v) else ('s+' if v > 0 else 's-')
            # dos sillas pegadas se leen como una sola mancha: se cuentan por su ancho a lo largo del lado
            along = (bw if horiz else bh) if side[0] == 'l' else (bh if horiz else bw)
            cnt[side] += max(1, min(3, int(round(along / 0.5)))) if side[0] == 'l' else 1
        cap = max(1, int((L + 0.05) // CHAIR_PITCH))
        want = max(cnt['l+'], cnt['l-'])
        if read and want > cap:
            # el folleto muestra más sillas de las que caben: la mesa es más larga de lo que se leyó
            newL = next((v for v in (1.20, 1.40, 1.60, 1.80, 2.00, 2.40) if v >= want * CHAIR_PITCH - 0.05), None)
            if newL is not None and newL - L <= 0.45:
                L = newL
                if horiz:
                    t['w'] = newL; hw = ctx.px(newL) / 2
                else:
                    t['d'] = newL; hh = ctx.px(newL) / 2
                cap = want
        if read:
            # los folletos muestran la mesa completa: en una mesa de 1,20 m o más, los lados largos van llenos
            n_long = cap if L >= 1.15 else min(cap, want)
            ends = [e for e in ('s+', 's-') if cnt[e]] if Sd >= 0.7 else []
        else:
            n_long = cap if L >= 1.15 else 1
            ends = []
        if n_long == 0 and not ends:
            n_long = 1
        H_long = ctx.px(Sd) / 2 + off                 # distancia del centro de la mesa al centro de la silla
        H_end = ctx.px(L) / 2 + off
        for sg in (1, -1):
            for k in range(n_long):
                f = ((k + 0.5) / n_long - 0.5) * ctx.px(L)
                seats.append((cx + f, cy + sg * H_long) if horiz else (cx + sg * H_long, cy + f))
        for e in ends:
            sg = 1 if e == 's+' else -1
            seats.append((cx + sg * H_end, cy) if horiz else (cx, cy + sg * H_end))
    others = [_rect(ctx, o) for o in items if o is not t and o['room'] == rid]
    for bx, by in seats:
        a = _ang_to(ctx, bx, by, cx, cy)
        if not t.get('round'):
            a = _snap_ang(a)
        it = dict(type='chair', cx=float(bx), cy=float(by), w=CHAIR_W, d=CHAIR_D, ang=float(a), room=rid, src='rule')
        r_ = _rect(ctx, it)
        if rl is not None:
            x0, y0, x1, y1 = int(r_[0]), int(r_[1]), int(math.ceil(r_[2])), int(math.ceil(r_[3]))
            sub = ctx.lab[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if sub.size == 0 or (sub == rl).mean() < 0.85:
                continue                              # contra un muro o fuera del espacio
        if sum(_inter(r_, o) for o in others) > 0.15 * _area(r_):
            continue                                  # encima de otro mueble
        items.append(it); others.append(r_)


# ---------------------------------------------------------------------------------------------- plantas
def _plants(ctx, rooms, items):
    hsv = cv2.cvtColor(ctx.im, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] >= 33) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 35)).astype(np.uint8)
    inside = np.zeros_like(g)
    for r in rooms.values():
        if not r.get('exterior') and r.get('type') != 'patio':
            inside |= (ctx.lab == r['label']).astype(np.uint8)
    g &= inside
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(g, 8)
    tables = [it for it in items if it['type'] in ('dining_table', 'coffee_table', 'desk', 'nightstand', 'tv_unit', 'misc')]
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        am = a * ctx.s ** 2
        if not (0.03 <= am <= 0.8):
            continue
        cx, cy = x + w / 2, y + h / 2
        if any(abs(cx - t['cx']) < ctx.px(t['w']) / 2 + 2 and abs(cy - t['cy']) < ctx.px(t['d']) / 2 + 2 for t in tables):
            continue                                   # matera sobre una mesa: no es una planta en el piso
        dia = float(np.clip(max(w, h) * ctx.s, 0.35, 0.8))
        rid = next((r['id'] for r in rooms.values() if ctx.lab[int(cy), int(cx)] == r['label']), None)
        out.append(dict(type='plant', cx=float(cx), cy=float(cy), w=dia, d=dia, ang=0.0, room=rid, src='color',
                        box=(x, y, x + w, y + h)))
    return out


# ---------------------------------------------------------------------------------------------- aparatos
def _fixture_items(ctx, objs, rooms):
    out = _fixture_items0(ctx, objs, rooms)
    for it, o in zip(out, [o for o in objs if o['kind'] == 'fixture' and o.get('fixture') and rooms.get(o['room']) is not None]):
        it['_box'] = _box(o)
        it.setdefault('box_wh', (o['w_m'], o['h_m']))
    return out


def _fixture_items0(ctx, objs, rooms):
    out = []
    for o in objs:
        f = o.get('fixture')
        if o['kind'] != 'fixture' or not f:
            continue
        r = rooms.get(o['room'])
        if r is None:
            continue
        rl = r['label']
        box = _box(o)
        dd = ctx.dists(rl, box, )
        wm, hm = o['w_m'], o['h_m']
        if f == 'toilet':
            sides = [sd for sd in 'NSEW' if (sd in 'NS') == (hm >= wm)] or list('NSEW')
            back = min(sides, key=lambda sd: dd[sd])
            it = _place(ctx, rl, box, back, 0.40, 0.70, f, o['room'], flush=0.4, dist=dd[back])
            it['cx'], it['cy'] = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            it['box_wh'] = (wm, hm)
            out.append(it)
        elif f == 'bathtub':
            sides = 'NS' if wm >= hm else 'EW'
            back = min(sides, key=lambda sd: dd[sd])
            L = _snap(max(wm, hm), (1.40, 1.50, 1.60, 1.70, 1.80))
            out.append(_place(ctx, rl, box, back, L, 0.75, f, o['room'], flush=0.4, dist=dd[back]))
        elif f == 'shower':
            back = min('NSEW', key=lambda sd: dd[sd])
            w, d = _orient_box(box, back, ctx)
            out.append(_place(ctx, rl, box, back, round(min(max(w, 0.8), 1.4) / 0.05) * 0.05,
                              round(min(max(d, 0.8), 1.2) / 0.05) * 0.05, f, o['room'], flush=0.4, dist=dd[back]))
        elif f == 'lavatory':
            back = min('NSEW', key=lambda sd: dd[sd])
            w, d = _orient_box(box, back, ctx)
            w = round(min(max(w, 0.45), 1.6) / 0.05) * 0.05
            out.append(_place(ctx, rl, box, back, w, 0.48, f, o['room'], flush=0.4, dist=dd[back]))
        elif f in ('cooktop', 'sink'):
            back = min('NSEW', key=lambda sd: dd[sd])
            w, d = _orient_box(box, back, ctx)
            if f == 'cooktop':
                w, d = 0.60, 0.52
            else:
                w = 0.80 if w >= 0.7 else 0.55
                d = 0.48
            it = _place(ctx, rl, box, back, w, d, f, o['room'], flush=0.75, dist=dd[back])
            # sobre el mesón: el respaldo queda a 0,05 m del muro (el mesón tiene ~0,60 m de fondo)
            if dd[back] <= 0.75:
                sh = ctx.px(0.05)
                if back == 'N':
                    it['cy'] += sh
                elif back == 'S':
                    it['cy'] -= sh
                elif back == 'W':
                    it['cx'] += sh
                else:
                    it['cx'] -= sh
            out.append(it)
        else:                                            # nevera / lavadora: caja contra el muro
            back = min('NSEW', key=lambda sd: dd[sd])
            if f == 'fridge':
                w = d = 0.70
            else:
                w = d = 0.60
            out.append(_place(ctx, rl, box, back, w, d, f, o['room'], flush=0.5, dist=dd[back]))
    return out


def _kitchen_fix(ctx, model, objs, rooms, items):
    """v0.4: estufa y lavaplatos sobre el mesón sin encimarse (≥ 0,30 m entre ellos); nevera y lavadora contra un
    muro, sin montarse en el mesón, en otra pieza ni en el barrido de una puerta."""
    s = ctx.s
    for rid, r in rooms.items():
        kit = [it for it in items if it['room'] == rid and it['type'] in ('cooktop', 'sink', 'fridge', 'washer')]
        if not kit:
            continue
        rl = r['label']
        zones = _door_zones(ctx, model, rl, rid)
        counters = []
        for o in objs:
            if o['room'] == rid and o['kind'] == 'casework':
                b = _box(o)
                counters.append(dict(type='counter', cx=(b[0] + b[2]) / 2, cy=(b[1] + b[3]) / 2,
                                     w=(b[2] - b[0]) * s, d=(b[3] - b[1]) * s, ang=0.0))
        top = [it for it in kit if it['type'] in ('cooktop', 'sink')]
        for a in top:
            for b in top:
                if a is b or a['type'] != 'cooktop':
                    continue
                ra, rb = _rect(ctx, a), _rect(ctx, b)
                horiz = a['ang'] in (0.0, 180.0)
                gap = (max(rb[0] - ra[2], ra[0] - rb[2]) if horiz else max(rb[1] - ra[3], ra[1] - rb[3])) * s
                if gap >= 0.30 or abs((a['cy'] - b['cy']) if horiz else (a['cx'] - b['cx'])) * s > 0.5:
                    continue
                # el lavaplatos se corre a lo largo del mesón, del lado donde ya estaba
                sgn = 1 if ((b['cx'] - a['cx']) if horiz else (b['cy'] - a['cy'])) >= 0 else -1
                need = ctx.px(0.30 - gap) if gap > -1 else 0
                if horiz:
                    b['cx'] += sgn * (need + 1)
                else:
                    b['cy'] += sgn * (need + 1)
        fixed = list(top)
        for it in [it for it in kit if it['type'] in ('fridge', 'washer')]:
            others = counters + [o for o in fixed if o is not it]
            r_ = _rect(ctx, it)
            hit = sum(_inter(r_, _rect(ctx, o)) for o in others) > 0.1 * _area(r_)
            door = sum(_inter(r_, z) for z in zones) > 0.2 * _area(r_)
            if hit or door:
                ok = False
                x0, y0, x1, y1 = r_
                dd = ctx.dists(rl, (int(x0), int(y0), int(max(x1, x0 + 1)), int(max(y1, y0 + 1))))
                for sd in sorted('NSEW', key=lambda q: dd[q]):
                    t = dict(it)
                    if _seat(ctx, t, sd, rl, others, zones, strict=True):
                        rr = _rect(ctx, t)
                        if math.dist((t['cx'], t['cy']), (it['cx'], it['cy'])) * s <= 1.5 and \
                                sum(_inter(rr, z) for z in zones) <= 0.2 * _area(rr):
                            t.pop('span', None)
                            it.update(t); ok = True
                            break
                if not ok and r.get('open_kitchen'):
                    items.remove(it)                      # en una sala abierta y sin sitio creíble: no se dibuja
                    continue
            fixed.append(it)


def _kitchen_retype(ctx, objs, rooms):
    """Estufa vs. nevera por apariencia: la estufa es oscura (vidrio) o tiene anillos; la nevera es clara."""
    gray = cv2.cvtColor(ctx.im, cv2.COLOR_BGR2GRAY)
    kfx = [o for o in objs if o.get('fixture') in ('cooktop', 'fridge')]
    if len(kfx) < 2:
        return
    def dark(o):
        sub = gray[o['y0']:o['y1'], o['x0']:o['x1']]
        return float((sub < 95).mean()) if sub.size else 0.0
    for rid in {o['room'] for o in kfx}:
        cook = [o for o in kfx if o['room'] == rid and o['fixture'] == 'cooktop']
        frid = [o for o in kfx if o['room'] == rid and o['fixture'] == 'fridge']
        if len(cook) == 1 and len(frid) == 1:
            c, f = cook[0], frid[0]
            if dark(f) >= 0.35 and dark(c) <= 0.15 and _dims(c)[0] >= 0.45:
                c['fixture'], f['fixture'] = 'fridge', 'cooktop'


# ---------------------------------------------------------------------------------------------- baños
WET = ('toilet', 'lavatory', 'shower', 'bathtub', 'washer')
STD = {'toilet': (0.40, 0.70), 'lavatory': (0.50, 0.45), 'shower': (0.90, 0.90), 'bathtub': (1.60, 0.75), 'washer': (0.60, 0.60)}


def _door_zones(ctx, model, rl, rid):
    """Zona de barrido de cada puerta del espacio (cuadrado del ancho de la hoja, del lado del espacio)."""
    zones = []
    for op in model.get('openings', []):
        if op['kind'] != 'door' or rid not in op.get('rooms', [rid]):
            continue
        x0, y0, x1, y1 = int(op['x0']), int(op['y0']), int(op['x1']), int(op['y1'])
        if op['orient'] == 'h':
            w = x1 - x0; cx = (x0 + x1) // 2
            up = 0 <= y0 - 3 < ctx.H and ctx.lab[y0 - 3, min(cx, ctx.W - 1)] == rl
            zones.append((x0, y0 - w, x1, y0) if up else (x0, y1, x1, y1 + w))
        else:
            w = y1 - y0; cy = (y0 + y1) // 2
            left = 0 <= x0 - 3 < ctx.W and ctx.lab[min(cy, ctx.H - 1), x0 - 3] == rl
            zones.append((x0 - w, y0, x0, y1) if left else (x1, y0, x1 + w, y1))
    return zones


def _span(ctx, rl, back, face, c):
    """Tramo libre (px) a lo largo del muro de respaldo, medido 2 px dentro del espacio, a partir de c."""
    if back in 'NS':
        y = int(face + (2 if back == 'N' else -3)); y = min(max(y, 0), ctx.H - 1)
        line = ctx.lab[y]
    else:
        x = int(face + (2 if back == 'W' else -3)); x = min(max(x, 0), ctx.W - 1)
        line = ctx.lab[:, x]
    c = int(min(max(c, 0), len(line) - 1))
    if line[c] != rl:
        return c, c
    a = c
    while a > 0 and line[a - 1] == rl:
        a -= 1
    b = c
    while b < len(line) - 1 and line[b + 1] == rl:
        b += 1
    return a, b + 1


def _face(ctx, rl, cx, cy, back, half):
    """Cara del muro en la dirección `back`, desde el centro de la pieza (mediana de 3 rayos)."""
    out = []
    for t in (-half, 0, half):
        x, y = int(round(cx + (t if back in 'NS' else 0))), int(round(cy + (t if back in 'EW' else 0)))
        x = min(max(x, 0), ctx.W - 1); y = min(max(y, 0), ctx.H - 1)
        if ctx.lab[y, x] != rl:
            continue
        dx, dy = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}[back]
        while 0 <= x + dx < ctx.W and 0 <= y + dy < ctx.H and ctx.lab[y + dy, x + dx] == rl:
            x += dx; y += dy
        out.append(y + (1 if back == 'S' else 0) if back in 'NS' else x + (1 if back == 'E' else 0))
    if not out:
        return None
    return float(np.median(out))


def _seat(ctx, it, back, rl, room_items, zones, along=None, strict=False):
    """Pone la pieza con el respaldo pegado al muro `back`, dentro del tramo libre y fuera de puertas y otras piezas.
    strict: no se acepta ninguna posición que choque con otra pieza (con 6 cm de holgura a los lados y, para inodoro
    y lavamanos, 0,45 m libres al frente); devuelve False en vez de encimarla."""
    s = ctx.s
    w, d = ctx.px(it['w']), ctx.px(it['d'])
    face = _face(ctx, rl, it['cx'], it['cy'], back, w / 3)
    if face is None:
        return False
    c = it['cx'] if back in 'NS' else it['cy']
    if along is not None:
        c = along
    a, b = _span(ctx, rl, back, face, c)
    if b - a < w * 0.9:
        return False
    sgn = 1 if back in 'NW' else -1
    depth_c = face + sgn * d / 2
    pad = ctx.px(0.06) if strict else 0.0
    fd = ctx.px(0.45)
    others = [_rect(ctx, o) for o in room_items if o is not it]

    def rect_at(cc):
        if back in 'NS':
            return (cc - w / 2, depth_c - d / 2, cc + w / 2, depth_c + d / 2)
        return (depth_c - d / 2, cc - w / 2, depth_c + d / 2, cc + w / 2)

    def front(r):
        if back == 'N':
            return (r[0], r[3], r[2], r[3] + fd)
        if back == 'S':
            return (r[0], r[1] - fd, r[2], r[1])
        if back == 'W':
            return (r[2], r[1], r[2] + fd, r[3])
        return (r[0] - fd, r[1], r[0], r[3])

    def bad(cc):
        """(choca con otra pieza, invade el barrido de una puerta). El barrido real es un cuarto de círculo:
        se tolera rozar la esquina del cuadrado."""
        r = rect_at(cc)
        rr = (r[0] - pad, r[1], r[2] + pad, r[3]) if back in 'NS' else (r[0], r[1] - pad, r[2], r[3] + pad)
        items_hit = sum(_inter(rr, o) for o in others) > 0.02 * w * d
        if strict and not items_hit and it['type'] in ('toilet', 'lavatory'):
            fr = front(r)
            items_hit = sum(_inter(fr, o) for o in others) > 0.2 * _area(fr)
        door_hit = sum(_inter(r, z) for z in zones) > 0.25 * w * d
        return items_hit, door_hit
    lo, hi = a + w / 2, b - w / 2
    best = min(np.linspace(lo, hi, max(2, int((hi - lo) / max(1.0, ctx.px(0.05))) + 1)),
               key=lambda cc: (*bad(cc), abs(cc - c)))
    if strict and bad(best)[0]:
        return False
    if back in 'NS':
        it['cx'], it['cy'] = float(best), float(depth_c)
    else:
        it['cx'], it['cy'] = float(depth_c), float(best)
    it['ang'] = ANG[back]
    it['span'] = (a, b)
    return True


def _room_box(ctx, rl):
    ys, xs = np.nonzero(ctx.lab == rl)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _shower_marks(ctx, rl, avoid=()):
    """Señales de ducha en la imagen del baño, fuera de las piezas que el modelo vio:
    - sifón: mancha redonda de 4-20 cm, aislada, que contrasta con el piso (oscura sobre piso claro o clara sobre
      piso oscuro), a más de 10 cm del muro; peso 1,5;
    - regadera vista desde arriba: disco claro de 20-30 cm a menos de 0,5 m de un muro; peso 2.
    -> [(x, y, peso)]"""
    bb = _room_box(ctx, rl)
    if bb is None:
        return []
    x0, y0, x1, y1 = bb
    s = ctx.s
    room = np.ascontiguousarray(ctx.lab[y0:y1, x0:x1] == rl)
    if room.sum() < 20:
        return []
    sub = ctx.im[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    v, sat = hsv[..., 2].astype(int), hsv[..., 1].astype(int)
    med = int(np.median(v[room]))
    inner = cv2.erode(room.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dist = cv2.distanceTransform(np.ascontiguousarray(room.astype(np.uint8)), cv2.DIST_L2, 5) * s
    ring = max(3, int(round(ctx.px(0.12))))
    out = []
    for pol in (-1, 1):
        msk = inner & ((v <= med - 30) if pol < 0 else ((v >= med + 30) & (sat <= 60)))
        n, lab_, st, cen = cv2.connectedComponentsWithStats(msk.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            dia = 2 * math.sqrt(a / math.pi) * s
            if not (0.04 <= dia <= 0.30) or max(w, h) / max(1, min(w, h)) > 1.5 or a / float(w * h) < 0.5:
                continue
            cx, cy = float(cen[i][0]), float(cen[i][1])
            X, Y = x0 + cx, y0 + cy
            if any(q[0] <= X <= q[2] and q[1] <= Y <= q[3] for q in avoid):
                continue
            ya, yb, xa, xb = max(0, int(cy) - ring), int(cy) + ring + 1, max(0, int(cx) - ring), int(cx) + ring + 1
            around = int(msk[ya:yb, xa:xb].sum()) - int((lab_[ya:yb, xa:xb] == i).sum())
            if around > 3:
                continue                                # parte de un texto, un patrón o una pieza
            dw = float(dist[int(cy), int(cx)])
            if pol > 0 and dia >= 0.20 and dw <= 0.5:
                out.append((X, Y, 2.0))                 # regadera
            elif dia <= 0.20 and dw >= 0.10:
                out.append((X, Y, 1.5))                 # sifón
    return out


def _wet_zone(ctx, r, zones, pieces, extra=()):
    """Ducha del baño cuando el modelo no la marcó, como lista de opciones de la más a la menos probable.
    Convención de folleto y de obra: un rectángulo de muro a muro en un extremo del baño (0,90 m de fondo; 1,00 m
    en baños largos). Se elige por señales en la imagen (regadera, sifón); sin señales, en un baño completo, el
    extremo más lejano de la puerta (supuesta). Si con ella no caben inodoro y lavamanos, la siguiente opción es
    un cuadrado de 0,90 m en la esquina de las señales. Nunca sobre las piezas que el modelo vio ni sobre el
    barrido de la puerta. -> [pieza 'shower', ...]"""
    rl, rid, s = r['label'], r['id'], ctx.s
    bb = _room_box(ctx, rl)
    if bb is None:
        return []
    bx0, by0, bx1, by1 = bb
    Wm, Hm = (bx1 - bx0) * s, (by1 - by0) * s
    prects = [_rect(ctx, p) for p in pieces]
    avoid = [(a[0] - ctx.px(0.05), a[1] - ctx.px(0.05), a[2] + ctx.px(0.05), a[3] + ctx.px(0.05)) for a in prects]
    marks = _shower_marks(ctx, rl, avoid) + list(extra)
    D = ctx.px(1.0 if max(Wm, Hm) >= 2.6 else 0.9)
    Q = ctx.px(0.9)
    strips, corners = [], []
    if Wm <= 2.2 and Hm - D * s >= 0.9:
        strips += [(bx0, by0, bx1, by0 + D), (bx0, by1 - D, bx1, by1)]
    if Hm <= 2.2 and Wm - D * s >= 0.9:
        strips += [(bx0, by0, bx0 + D, by1), (bx1 - D, by0, bx1, by1)]
    if min(Wm, Hm) >= 1.4:
        corners = [(bx0, by0, bx0 + Q, by0 + Q), (bx1 - Q, by0, bx1, by0 + Q),
                   (bx0, by1 - Q, bx0 + Q, by1), (bx1 - Q, by1 - Q, bx1, by1)]
    dc = [((z[0] + z[2]) / 2, (z[1] + z[3]) / 2) for z in zones]
    diag = math.hypot(bx1 - bx0, by1 - by0)

    hsv = cv2.cvtColor(ctx.im, cv2.COLOR_BGR2HSV)
    rv = hsv[..., 2][ctx.lab == rl]
    med = int(np.median(rv)) if rv.size else 200

    def rate(z):
        zi = (int(z[0]), int(z[1]), int(math.ceil(z[2])), int(math.ceil(z[3])))
        sub = ctx.lab[max(0, zi[1]):zi[3], max(0, zi[0]):zi[2]]
        cover = float((sub == rl).mean()) if sub.size else 0.0
        hv = hsv[max(0, zi[1]):zi[3], max(0, zi[0]):zi[2]]
        white = float(((hv[..., 2] >= max(med + 10, 235)) & (hv[..., 1] <= 25) & (sub == rl)).mean()) if sub.size else 0.0
        L = max(z[2] - z[0], z[3] - z[1]) * s
        tub = white >= 0.45 and 1.5 <= L <= 2.2         # tina: porcelana blanca de muro a muro
        g = ctx.px(0.1)
        ev = sum(wt for x, y, wt in marks if z[0] - g <= x <= z[2] + g and z[1] - g <= y <= z[3] + g)
        hit = sum(_inter(z, p) / max(_area(p), 1.0) for p in prects)
        door = sum(_inter(z, q) for q in zones) / max(_area(z), 1.0)
        far = min((math.dist(((z[0] + z[2]) / 2, (z[1] + z[3]) / 2), c) for c in dc), default=diag / 2) / max(diag, 1.0)
        if tub:
            ev += 3.0; hit = 0.0                         # lo que el modelo vio dentro de la tina es la tina
        return dict(z=z, cover=cover, ev=ev, hit=hit, door=door, tub=tub, score=3 * ev - 4 * hit - 6 * door + far)
    out = []
    full = r['area'] >= 2.3 and max(Wm, Hm) >= 1.7
    for kind, zs in (('strip', strips), ('corner', corners)):
        rs = sorted((rate(z) for z in zs), key=lambda q: -q['score'])
        for q in rs:
            if q['cover'] < 0.75 or q['door'] >= 0.2 or q['hit'] >= 0.5:
                continue
            if q['ev'] >= 1.0:
                src = 'image'
            elif kind == 'strip' and not marks and full and q['hit'] < 0.3 and q['door'] < 0.15:
                src = 'assumed'
            else:
                continue
            z = q['z']
            if kind == 'strip' and q['tub']:
                # tina de 0,75 m de fondo pegada al muro del extremo
                dz = ctx.px(0.75)
                if (z[2] - z[0]) >= (z[3] - z[1]):
                    z = (z[0], z[1], z[2], z[1] + dz) if z[1] <= by0 + 1 else (z[0], z[3] - dz, z[2], z[3])
                else:
                    z = (z[0], z[1], z[0] + dz, z[3]) if z[0] <= bx0 + 1 else (z[2] - dz, z[1], z[2], z[3])
                L = max(z[2] - z[0], z[3] - z[1]) * s
                horiz = (z[2] - z[0]) >= (z[3] - z[1])
                out.append(dict(type='bathtub', cx=(z[0] + z[2]) / 2, cy=(z[1] + z[3]) / 2, w=round(L, 2), d=0.75,
                                ang=0.0 if horiz else 90.0, room=rid, src='image'))
                continue
            out.append(dict(type='shower', cx=(z[0] + z[2]) / 2, cy=(z[1] + z[3]) / 2, w=round((z[2] - z[0]) * s, 2),
                            d=round((z[3] - z[1]) * s, 2), ang=0.0, room=rid, src=src))
            if src == 'assumed' or kind == 'corner':
                break                                    # una sola opción supuesta / de esquina
    return out


def _seat_wet(ctx, it, rl, placed, zones, strict):
    """Respaldo de la pieza húmeda contra el muro más cercano (tanque del inodoro, poceta del lavamanos)."""
    s = ctx.s
    x0, y0, x1, y1 = _rect(ctx, it)
    box = (int(x0), int(y0), int(max(x1, x0 + 1)), int(max(y1, y0 + 1)))
    dd = ctx.dists(rl, box)
    bw, bh = it.get('box_wh', ((x1 - x0) * s, (y1 - y0) * s))
    if it['type'] == 'bathtub':                       # la tina va de lado contra el muro
        sides = 'NS' if bw >= bh else 'EW'
    else:
        sides = 'NSEW'
    back = min(sides, key=lambda sd: dd[sd])
    if it['type'] == 'toilet' and max(bw, bh) / max(min(bw, bh), 1e-3) >= 1.25:
        # inodoro leído alargado: el tanque va en un extremo corto, si hay muro cerca
        ends = 'NS' if bh > bw else 'EW'
        e = min(ends, key=lambda sd: dd[sd])
        if dd[e] <= 0.6:
            back = e
    order = [back] + [sd for sd in sorted(sides, key=lambda sd: dd[sd]) if sd != back]
    if strict:
        order = [sd for sd in order if dd[sd] <= 0.9] or order[:1]
    for sd in order:
        if _seat(ctx, it, sd, rl, placed, zones, strict=strict):
            return True
    return False


def _bath_pair(ctx, model, r, zones, placed, small, items):
    """Inodoro y lavamanos adosados a la misma pared, fuera de la ducha y del barrido de la puerta (lo usual en
    obra: un solo muro húmedo). El inodoro del lado de la ducha, el lavamanos del lado de la puerta, salvo que el
    modelo los haya visto en el otro orden. Si los dos caben en un muro, se ubican ahí (los que faltan, supuestos)
    y se devuelven las piezas que quedan por ubicar; si no, se devuelven todas para ubicarlas una a una."""
    rl, rid, s = r['label'], r['id'], ctx.s
    toil = [o for o in small if o['type'] == 'toilet']
    lavs = [o for o in small if o['type'] == 'lavatory']
    rest = [o for o in small if o['type'] not in ('toilet', 'lavatory')]
    if len(toil) > 1 or len(lavs) > 1 or r['area'] < 1.4:
        return small
    t0 = toil[0] if toil else dict(type='toilet', cx=0.0, cy=0.0, w=STD['toilet'][0], d=STD['toilet'][1], ang=0.0,
                                   room=rid, src='assumed')
    l0 = lavs[0] if lavs else dict(type='lavatory', cx=0.0, cy=0.0, w=STD['lavatory'][0], d=STD['lavatory'][1], ang=0.0,
                                   room=rid, src='assumed')
    t0['w'], t0['d'] = STD['toilet']
    if toil and lavs:
        t1, l1 = dict(t0), dict(l0)
        if _seat_wet(ctx, t1, rl, list(placed), zones, strict=True) and \
                _seat_wet(ctx, l1, rl, list(placed) + [t1], zones, strict=True) and \
                math.dist((t1['cx'], t1['cy']), (t0['cx'], t0['cy'])) * s <= 0.35 and \
                math.dist((l1['cx'], l1['cy']), (l0['cx'], l0['cy'])) * s <= 0.35:
            return small                                  # el modelo los vio bien puestos: se respetan
    bb = _room_box(ctx, rl)
    if bb is None:
        return small
    bx0, by0, bx1, by1 = bb
    sh = next((o for o in placed if o['type'] in ('shower', 'bathtub')), None)
    shr = _rect(ctx, sh) if sh else None
    # sin piezas vistas: lo que el modelo marcó como aparato o mueble en el baño (fuera de la ducha) indica el muro
    ev = np.isin(ctx.mask, (AP, MS, MU)) & (ctx.lab == rl)
    if shr is not None:
        ev[max(0, int(shr[1])):int(math.ceil(shr[3])), max(0, int(shr[0])):int(math.ceil(shr[2]))] = False
    ey, ex = np.nonzero(ev)
    ev_c = (float(ex.mean()), float(ey.mean())) if ex.size > 15 else None
    dc = [((z[0] + z[2]) / 2, (z[1] + z[3]) / 2) for z in zones]
    seen = [o for o in (toil + lavs)]
    best = None
    for sd in 'NSEW':
        along_lo, along_hi = (bx0, bx1) if sd in 'NS' else (by0, by1)
        # extremo «ducha» y extremo «puerta» a lo largo de este muro
        if shr is not None:
            sc = (shr[0] + shr[2]) / 2 if sd in 'NS' else (shr[1] + shr[3]) / 2
            if sd in 'NS' and (shr[2] - shr[0]) > 0.8 * (bx1 - bx0) or sd in 'EW' and (shr[3] - shr[1]) > 0.8 * (by1 - by0):
                continue                                   # la ducha ocupa todo este muro
        else:
            sc = None
        if sc is None and dc:
            dmid = dc[0][0] if sd in 'NS' else dc[0][1]
            sc = along_hi if dmid - along_lo < along_hi - dmid else along_lo
        if sc is None:
            sc = along_lo
        toward = 1 if sc > (along_lo + along_hi) / 2 else -1   # dirección hacia la ducha
        if shr is not None:
            edge = (shr[0] if toward > 0 else shr[2]) if sd in 'NS' else (shr[1] if toward > 0 else shr[3])
        else:
            edge = along_hi if toward > 0 else along_lo
        for first in ('toilet', 'lavatory'):
            # la primera pieza junto a la ducha (o al fondo), la segunda a continuación hacia la puerta
            p1 = edge - toward * ctx.px(0.30)
            p2 = p1 - toward * ctx.px(0.62)
            t_al, l_al = (p1, p2) if first == 'toilet' else (p2, p1)
            if toil and lavs:                             # orden visto por el modelo en este muro: se respeta
                a_t = toil[0]['cx'] if sd in 'NS' else toil[0]['cy']
                a_l = lavs[0]['cx'] if sd in 'NS' else lavs[0]['cy']
                if (a_l - a_t) * (l_al - t_al) < 0:
                    continue
            t = dict(t0); l_ = dict(l0)
            for q in (t, l_):
                if sd == 'N':
                    q['cy'] = by0 + ctx.px(q['d']) / 2
                elif sd == 'S':
                    q['cy'] = by1 - ctx.px(q['d']) / 2
                elif sd == 'W':
                    q['cx'] = bx0 + ctx.px(q['d']) / 2
                else:
                    q['cx'] = bx1 - ctx.px(q['d']) / 2
                if sd in 'NS':
                    q['cy'] = float(np.clip(q['cy'], by0 + 1, by1 - 2))
                    q['cx'] = float((bx0 + bx1) / 2)
                else:
                    q['cx'] = float(np.clip(q['cx'], bx0 + 1, bx1 - 2))
                    q['cy'] = float((by0 + by1) / 2)
            base = list(placed)
            q1, a1, q2, a2 = (t, t_al, l_, l_al) if first == 'toilet' else (l_, l_al, t, t_al)
            if not _seat(ctx, q1, sd, rl, base, zones, along=float(a1), strict=True):
                continue
            if not _seat(ctx, q2, sd, rl, base + [q1], zones, along=float(a2), strict=True):
                continue
            door = sum(_inter(_rect(ctx, q), z) for q in (t, l_) for z in zones)
            if door > 0.1 * (_area(_rect(ctx, t)) + _area(_rect(ctx, l_))):
                continue
            # preferir el muro donde el modelo vio las piezas, el inodoro junto a la ducha y lo que el modelo marcó
            moved = sum(math.dist((q['cx'], q['cy']), (o['cx'], o['cy'])) * s + (0.5 if q['ang'] != o['ang'] else 0.0)
                        for q, o in ((t, t0), (l_, l0)) if o in seen)
            score = -moved - 0.3 * (first != 'toilet') - 0.5 * abs(((q1['cx'] if sd in 'NS' else q1['cy']) - a1)) * s
            if not seen and ev_c is not None:
                score -= math.dist(((t['cx'] + l_['cx']) / 2, (t['cy'] + l_['cy']) / 2), ev_c) * s
            if best is None or score > best[0]:
                best = (score, t, l_)
    if best is None:
        return small
    _, t, l_ = best
    for q, o in ((t, t0), (l_, l0)):
        q.pop('span', None); q.pop('box_wh', None)
        o.clear(); o.update(q)
        if o not in items:
            items.append(o)
        placed.append(o)
    return rest


def _place_small(ctx, model, r, rl, zones, placed, small, items):
    """Inodoro y lavamanos (juntos en un muro si se puede) y lavadoras; devuelve cuántas piezas leídas no cupieron."""
    dropped = 0
    rest = _bath_pair(ctx, model, r, zones, placed, list(small), items) if r.get('type') == 'bath' else list(small)
    for it in rest:
        if it['type'] == 'toilet':
            it['w'], it['d'] = STD['toilet']
        ok = _seat_wet(ctx, it, rl, placed, zones, strict=True)
        if not ok:                                       # no cabe donde se leyó: el sitio libre más cercano
            alt = _assume(ctx, model, r, it['type'], placed, zones, target=(it['cx'], it['cy']), w=it['w'])
            if alt is not None:
                it.update(cx=alt['cx'], cy=alt['cy'], ang=alt['ang']); ok = True
        if not ok:
            _seat_wet(ctx, it, rl, placed, zones, strict=False)
            r_ = _rect(ctx, it)
            if any(_inter(r_, _rect(ctx, o)) > 0.3 * _area(r_) for o in placed):
                if it in items:
                    items.remove(it)                     # sin sitio: mejor no dibujarla encima de otra
                it.pop('span', None)
                dropped += 1
                continue
        it.pop('span', None)
        placed.append(it)
    return dropped


def _missing(ctx, model, r, placed, zones, need):
    """Piezas supuestas que faltan (inodoro, lavamanos), ubicadas una tras otra para que quepan juntas.
    -> lista (vacía si no falta nada) o None si no caben."""
    have = {o['type'] for o in placed}
    todo = [t for t in ('toilet', 'lavatory') if t in need and t not in have]
    if not todo:
        return []
    for seq in (todo, todo[::-1]):
        p2, out = list(placed), []
        for t in seq:
            it = _assume(ctx, model, r, t, p2, zones)
            if it is None:
                break
            p2.append(it); out.append(it)
        else:
            return out
    return None


def _wet_fixtures(ctx, model, rooms, items):
    """Aparatos sanitarios y de ropas: el respaldo (tanque del inodoro, poceta del lavamanos) siempre contra el muro.
    v0.4: primero la tina o la ducha (si el modelo no la vio, se busca en la imagen o se supone lejos de la puerta);
    después inodoro y lavamanos sin montarse en ella, en la puerta ni entre sí."""
    s = ctx.s
    order = {'bathtub': 0, 'shower': 1, 'toilet': 2, 'lavatory': 3, 'washer': 4}
    for rid, r in rooms.items():
        rl = r['label']
        wet = [it for it in items if it['room'] == rid and it['type'] in WET]
        if not wet and r.get('type') != 'bath':
            continue
        zones = _door_zones(ctx, model, rl, rid)
        placed = []
        for it in sorted([o for o in wet if o['type'] in ('bathtub', 'shower')], key=lambda o: order[o['type']]):
            if not _seat_wet(ctx, it, rl, placed, zones, strict=False):
                pass
            if it['type'] == 'bathtub' and 'span' in it:     # de muro a muro si el tramo se parece al largo
                a, b = it['span']
                L = (b - a) * s
                if it['w'] - 0.15 <= L <= it['w'] + 0.45 and 1.3 <= L <= 1.9:
                    it['w'] = round(L, 2)
                    if it['ang'] in (0.0, 180.0):
                        it['cx'] = (a + b) / 2.0
                    else:
                        it['cy'] = (a + b) / 2.0
            if it['type'] == 'shower' and 'span' in it:      # ducha en la esquina más cercana
                a, b = it['span']
                c = it['cx'] if it['ang'] in (0.0, 180.0) else it['cy']
                half = ctx.px(it['w']) / 2
                if c - half - a <= ctx.px(0.45):
                    c = a + half
                elif b - (c + half) <= ctx.px(0.45):
                    c = b - half
                if it['ang'] in (0.0, 180.0):
                    it['cx'] = c
                else:
                    it['cy'] = c
            it.pop('span', None); it.pop('box_wh', None)
            placed.append(it)
        small = sorted([o for o in wet if o['type'] not in ('bathtub', 'shower')], key=lambda o: order[o['type']])
        heads = []
        if r.get('type') == 'bath':
            # un disco pequeño y redondo junto al muro que el modelo llamó lavamanos o inodoro es la regadera
            for it in list(small):
                bw, bh = it.get('box_wh', (it['w'], it['d']))
                twin = [o for o in small if o['type'] == it['type'] and o is not it]
                if it['type'] in ('toilet', 'lavatory') and max(bw, bh) <= 0.36 and max(bw, bh) / max(min(bw, bh), 1e-3) <= 1.4 \
                        and '_box' in it and any(max(o.get('box_wh', (o['w'], o['d']))) >= 0.45 for o in twin):
                    b = it['_box']
                    heads.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 2.0))
                    small.remove(it); items.remove(it)
        options = [None]
        if r.get('type') == 'bath' and not placed:
            for it in small:                                 # posición leída, antes de moverla
                if it['type'] == 'toilet':
                    it['w'], it['d'] = STD['toilet']
            options = _wet_zone(ctx, r, zones, small, heads) + [None]
        snap = [dict(o) for o in small]
        base0 = list(placed)
        before = set(id(o) for o in items)
        need = {'toilet', 'lavatory'} if r.get('type') == 'bath' else set()
        for k, sh in enumerate(options):
            if k:                                            # deshacer el intento anterior
                items[:] = [o for o in items if id(o) in before or any(o is q for q in small)]
                for o, o0 in zip(small, snap):
                    o.clear(); o.update(o0)
                    if o not in items:
                        items.append(o)
            placed = list(base0)
            if sh is not None:
                items.append(sh); placed.append(sh)
                if sh['type'] == 'bathtub':
                    tr = _rect(ctx, sh)
                    for o in [o for o in small if tr[0] <= o['cx'] <= tr[2] and tr[1] <= o['cy'] <= tr[3]]:
                        small.remove(o)
                        if o in items:
                            items.remove(o)
            dropped = _place_small(ctx, model, r, rl, zones, placed, small, items)
            fits = _missing(ctx, model, r, placed, zones, need) is not None
            if sh is None or (not dropped and fits):
                break
        # todo baño tiene inodoro y lavamanos: si el modelo no los vio, se ponen contra un muro libre (supuestos)
        if r.get('type') == 'bath' and r['area'] >= 1.4:
            add = _missing(ctx, model, r, placed, zones, need) or []
            for it in add:
                items.append(it); placed.append(it)


def _assume(ctx, model, r, typ, placed, zones, target=None, w=None):
    """Pieza supuesta: recorre los muros del espacio y elige el sitio libre más cercano a `target` (por defecto, lo
    que el modelo marcó). No se acepta un sitio que choque con otra pieza ni que tape una puerta."""
    rl, rid = r['label'], r['id']
    sw, sd = STD[typ]
    if w is not None:
        sw = w
    ys, xs = np.nonzero(ctx.lab == rl)
    if xs.size == 0:
        return None
    if target is None:
        ev = np.isin(ctx.mask, (AP, MS, MU)) & (ctx.lab == rl)
        ey, ex = np.nonzero(ev)
        target = (float(ex.mean()), float(ey.mean())) if ex.size > 20 else (float(xs.mean()), float(ys.mean()))
    best = None
    for back in 'NSEW':
        for along in (None, 0.25, 0.75):
            it = dict(type=typ, cx=float(xs.mean()), cy=float(ys.mean()), w=sw, d=sd, ang=ANG[back], room=rid, src='assumed')
            # arranca pegado a ese muro, en línea con el objetivo (o a un cuarto / tres cuartos del muro)
            if back == 'N':
                it['cy'] = float(ys.min()) + ctx.px(sd) / 2
            elif back == 'S':
                it['cy'] = float(ys.max()) - ctx.px(sd) / 2
            elif back == 'W':
                it['cx'] = float(xs.min()) + ctx.px(sd) / 2
            else:
                it['cx'] = float(xs.max()) - ctx.px(sd) / 2
            if along is None:
                al = target[0] if back in 'NS' else target[1]
            else:
                al = (xs.min() + along * (xs.max() - xs.min())) if back in 'NS' else (ys.min() + along * (ys.max() - ys.min()))
            if not _seat(ctx, it, back, rl, placed, zones, along=float(al), strict=True):
                continue
            r_ = _rect(ctx, it)
            if sum(_inter(r_, z) for z in zones) > 0.25 * _area(r_):
                continue
            it.pop('span', None)
            dist = math.dist((it['cx'], it['cy']), target)
            if best is None or dist < best[0]:
                best = (dist, it)
    return best[1] if best else None


# ---------------------------------------------------------------------------------------------- principal
def _rect(ctx, it):
    """Caja (x0, y0, x1, y1) en píxeles de una pieza ortogonal."""
    a = round(it['ang'] / 90.0) % 2
    w, d = ctx.px(it['w']), ctx.px(it['d'])
    hx, hy = (w / 2, d / 2) if a == 0 else (d / 2, w / 2)
    return (it['cx'] - hx, it['cy'] - hy, it['cx'] + hx, it['cy'] + hy)


def _fit_beds(ctx, items):
    """Una cama no se monta sobre la mesa de noche, el escritorio o la silla: se angosta (ancho de catálogo)."""
    for b in [it for it in items if it['type'] == 'bed']:
        others = [it for it in items if it is not b and it['room'] == b['room'] and it['type'] in
                  ('nightstand', 'desk', 'chair', 'misc', 'bed')]
        for _ in range(4):
            bx = _rect(ctx, b)
            hit = [o for o in others if _inter(bx, _rect(ctx, o)) > 0.04 * _area(bx)]
            if not hit:
                break
            smaller = [w for w in BED_W if w < b['w'] - 1e-6]
            if not smaller:
                break
            o = max(hit, key=lambda o: _inter(bx, _rect(ctx, o)))
            ob = _rect(ctx, o)
            nw = smaller[-1]
            sh = ctx.px(b['w'] - nw) / 2
            lateral_x = round(b['ang'] / 90.0) % 2 == 0          # el ancho de la cama va en X de la imagen
            if lateral_x:
                b['cx'] += sh if (ob[0] + ob[2]) / 2 < b['cx'] else -sh
            else:
                b['cy'] += sh if (ob[1] + ob[3]) / 2 < b['cy'] else -sh
            b['w'] = nw
            b['d'] = 1.90 if nw <= 1.0 else 2.00


def _prune_vanities(objs):
    """Baños: un «mesón» solo es mueble de lavamanos si tiene el lavamanos encima o al lado."""
    for rid in {o['room'] for o in objs if o.get('casework') == 'vanity'}:
        van = [o for o in objs if o['room'] == rid and o.get('casework') == 'vanity']
        lav = [o for o in objs if o['room'] == rid and o.get('fixture') == 'lavatory']
        keep = [v for v in van if any(_gap(_box(v), _box(l), 1.0) <= 6 for l in lav)]
        if not keep and van:
            keep = [max(van, key=lambda o: o['w_m'] * o['h_m'])] if not lav else []
        for v in van:
            if v not in keep:
                objs.remove(v)


def _stray_counters(ctx, objs, rooms):
    """En la sala-comedor, un «mesón» lejos de la estufa, el lavaplatos y el rótulo de cocina es un mueble."""
    s = ctx.s
    for rid, r in rooms.items():
        if not (r.get('open_kitchen') or r.get('type') == 'living'):
            continue
        ev = [_center(_box(o)) for o in objs if o['room'] == rid and o.get('fixture') in ('cooktop', 'sink')]
        ev += [(t['mx'], t['my']) for t in r.get('texts', []) if t.get('type') == 'kitchen']
        for o in objs:
            if o['room'] != rid or o['kind'] != 'casework' or o.get('casework') not in ('counter', 'island / bar'):
                continue
            c = _center(_box(o))
            lo, hi = _dims(o)
            if not (0.5 <= lo and hi <= 2.4 and o['w_m'] * o['h_m'] <= 2.6 and o['fill'] >= 0.6):
                continue
            if not ev or min(math.dist(c, e) for e in ev) * s > 2.2:
                o['kind'] = 'furniture'; o.pop('casework', None); o['was_counter'] = True


def layout(image, mask, model, objs, rooms):
    ctx = _Ctx(image, mask, model)
    s = ctx.s
    _kitchen_retype(ctx, objs, rooms)
    _prune_vanities(objs)
    _stray_counters(ctx, objs, rooms)
    items = _fixture_items(ctx, objs, rooms)
    _kitchen_fix(ctx, model, objs, rooms, items)
    islands = [o for o in objs if o['kind'] == 'casework' and o.get('casework') in ('island / bar', 'counter')
               and _dims(o)[1] >= 0.9]
    for rid, r in rooms.items():
        if r.get('exterior') or r.get('type') in ('patio', 'bath', 'laundry', 'closet'):
            continue
        cands = [dict(box=_box(o), w_m=o['w_m'], h_m=o['h_m'], fill=o['fill'], was_counter=o.get('was_counter', False))
                 for o in objs if o['room'] == rid and o['kind'] == 'furniture']
        cands = _merge(cands, s)
        for c in cands:
            c['fill'] = c.get('fill', 0.7)
        t = r.get('type')
        used = set()
        if t in ('bed', 'study') or any(_dims(c)[0] >= 0.8 and _dims(c)[1] >= 1.45 and c['fill'] >= 0.6 for c in cands) \
                and t not in ('living', 'kitchen'):
            used |= _bedrooms(ctx, r, cands, items)
        if t in ('living', 'hall', 'kitchen') or r.get('open_kitchen'):
            isl = [o for o in islands if o['room'] == rid]
            used |= _living(ctx, r, [c for c in cands if id(c) not in used], items, isl)
        for c in cands:                                   # lo que queda: mueble genérico si tiene tamaño
            if id(c) in used:
                continue
            lo, hi = _dims(c)
            if lo >= 0.4 and c['w_m'] * c['h_m'] >= 0.2 and hi <= 2.6 and c['fill'] >= 0.45:
                cx, cy = _center(c['box'])
                if any(it['room'] == rid and abs(cx - it['cx']) < ctx.px(it['w'] + it['d']) / 2 and
                       abs(cy - it['cy']) < ctx.px(it['w'] + it['d']) / 2 for it in items):
                    continue
                it = _free('misc', c['box'], c['w_m'], c['h_m'], 0, rid)
                items.append(it)
    plants = _plants(ctx, rooms, items)
    # una planta no es un mueble genérico: lo que la planta cubre se descarta
    keep = []
    for it in items:
        if it['type'] == 'misc' and any(abs(it['cx'] - p['cx']) < ctx.px(p['w']) / 2 + 3 and
                                        abs(it['cy'] - p['cy']) < ctx.px(p['d']) / 2 + 3 for p in plants):
            continue
        keep.append(it)
    for p in plants:
        p.pop('box', None)
    out = keep + plants
    _fit_beds(ctx, out)
    _wet_fixtures(ctx, model, rooms, out)
    for it in out:
        it.pop('_box', None); it.pop('box_wh', None); it.pop('span', None)
    return out
