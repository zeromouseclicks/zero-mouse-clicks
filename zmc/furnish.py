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
        _chairs(ctx, t, chairs, items, rid)
        break
    return used


def _in_front(it, x, y, ctx):
    """¿(x, y) está del lado del frente de la pieza?"""
    a = math.radians(it['ang'])
    fx, fy = math.sin(a), -math.cos(a)            # frente en coordenadas de lámina (Y arriba)
    vx, vy = (x - it['cx']), -(y - it['cy'])
    return fx * vx + fy * vy > 0


def _chairs(ctx, t, boxes, items, rid):
    """Sillas alrededor de la mesa: donde se leyeron; si no se leyó ninguna, las de costumbre."""
    s = ctx.s
    cx, cy = t['cx'], t['cy']
    hw, hh = ctx.px(t['w']) / 2, ctx.px(t['d']) / 2
    seats = []
    for b in boxes:
        bx, by = _center(b)
        if math.dist((bx, by), (cx, cy)) * s < 0.25:
            continue
        a = _ang_to(ctx, bx, by, cx, cy)
        seats.append((bx, by, a))
    # simetría: las mesas de folleto casi siempre tienen sillas a ambos lados
    for bx, by, a in list(seats):
        if t.get('round'):
            m = (2 * cx - bx, 2 * cy - by)
        elif t['w'] >= t['d']:
            m = (bx, 2 * cy - by)
        else:
            m = (2 * cx - bx, by)
        if all(math.dist(m, (q[0], q[1])) * s >= 0.38 for q in seats):
            seats.append((m[0], m[1], None))
    if not seats:                                  # mesa sin sillas leídas: 4 (o 6 si es larga)
        long_ = max(t['w'], t['d'])
        n_side = 3 if long_ >= 1.75 else 2 if long_ >= 1.15 else 1
        horiz = t['w'] >= t['d']
        for sgn in (-1, 1):
            for k in range(n_side):
                f = (k + 0.5) / n_side - 0.5
                if horiz:
                    seats.append((cx + f * 2 * hw, cy + sgn * (hh + ctx.px(0.28)), None))
                else:
                    seats.append((cx + sgn * (hw + ctx.px(0.28)), cy + f * 2 * hh, None))
        if t.get('round') or n_side == 1:
            seats = [(cx + dx * (hw + ctx.px(0.28)), cy + dy * (hh + ctx.px(0.28)), None)
                     for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))]
    out = []
    for bx, by, a in seats:
        # la silla queda medio metida bajo la mesa: centro a 0,2 m del borde
        if t.get('round'):
            ang = math.atan2(by - cy, bx - cx)
            rr = hw + ctx.px(0.20)
            bx, by = cx + rr * math.cos(ang), cy + rr * math.sin(ang)
        else:
            dx, dy = bx - cx, by - cy
            if abs(dx) / max(hw, 1) > abs(dy) / max(hh, 1):
                bx = cx + math.copysign(hw + ctx.px(0.20), dx); by = float(np.clip(by, cy - hh + ctx.px(0.25), cy + hh - ctx.px(0.25)))
            else:
                by = cy + math.copysign(hh + ctx.px(0.20), dy); bx = float(np.clip(bx, cx - hw + ctx.px(0.25), cx + hw - ctx.px(0.25)))
        a = _ang_to(ctx, bx, by, cx, cy)
        if not t.get('round'):
            a = _snap_ang(a)
        if any(math.dist((bx, by), (o[0], o[1])) * s < 0.38 for o in out):
            continue
        out.append((bx, by, a))
    for bx, by, a in out[:10]:
        items.append(dict(type='chair', cx=float(bx), cy=float(by), w=0.46, d=0.46, ang=float(a), room=rid, src='rule'))


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


def _seat(ctx, it, back, rl, room_items, zones, along=None):
    """Pone la pieza con el respaldo pegado al muro `back`, dentro del tramo libre y fuera de puertas y otras piezas."""
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

    def rect_at(cc):
        if back in 'NS':
            return (cc - w / 2, depth_c - d / 2, cc + w / 2, depth_c + d / 2)
        return (depth_c - d / 2, cc - w / 2, depth_c + d / 2, cc + w / 2)

    def bad(cc):
        """(choca con otra pieza, invade el barrido de una puerta). El barrido real es un cuarto de círculo:
        se tolera rozar la esquina del cuadrado."""
        r = rect_at(cc)
        items_hit = sum(_inter(r, _rect(ctx, o)) for o in room_items if o is not it) > 0.02 * w * d
        door_hit = sum(_inter(r, z) for z in zones) > 0.25 * w * d
        return items_hit, door_hit
    lo, hi = a + w / 2, b - w / 2
    best = min(np.linspace(lo, hi, max(2, int((hi - lo) / max(1.0, ctx.px(0.05))) + 1)),
               key=lambda cc: (*bad(cc), abs(cc - c)))
    if back in 'NS':
        it['cx'], it['cy'] = float(best), float(depth_c)
    else:
        it['cx'], it['cy'] = float(depth_c), float(best)
    it['ang'] = ANG[back]
    it['span'] = (a, b)
    return True


def _wet_fixtures(ctx, model, rooms, items):
    """Aparatos sanitarios y de ropas: el respaldo (tanque del inodoro, poceta del lavamanos) siempre contra el muro."""
    s = ctx.s
    for rid, r in rooms.items():
        rl = r['label']
        wet = [it for it in items if it['room'] == rid and it['type'] in WET]
        if not wet and r.get('type') != 'bath':
            continue
        zones = _door_zones(ctx, model, rl, rid)
        placed = []
        order = {'bathtub': 0, 'shower': 1, 'toilet': 2, 'lavatory': 3, 'washer': 4}
        for it in sorted(wet, key=lambda o: order[o['type']]):
            sw, sd = STD[it['type']]
            if it['type'] == 'toilet':
                it['w'], it['d'] = sw, sd
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
            if not _seat(ctx, it, back, rl, placed, zones):
                for alt in sorted(sides, key=lambda sd: dd[sd])[1:]:
                    if _seat(ctx, it, alt, rl, placed, zones):
                        break
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
        # todo baño tiene inodoro y lavamanos: si el modelo no los vio, se ponen contra un muro libre (supuestos)
        if r.get('type') == 'bath' and r['area'] >= 1.4:
            for typ in ('toilet', 'lavatory'):
                if any(o['type'] == typ for o in placed):
                    continue
                it = _assume(ctx, model, r, typ, placed, zones)
                if it:
                    items.append(it); placed.append(it)


def _assume(ctx, model, r, typ, placed, zones):
    """Pieza supuesta: recorre los muros del espacio y elige el sitio libre más cercano a lo que el modelo marcó."""
    rl, rid = r['label'], r['id']
    sw, sd = STD[typ]
    ys, xs = np.nonzero(ctx.lab == rl)
    if xs.size == 0:
        return None
    ev = np.isin(ctx.mask, (AP, MS, MU)) & (ctx.lab == rl)
    ey, ex = np.nonzero(ev)
    target = (float(ex.mean()), float(ey.mean())) if ex.size > 20 else (float(xs.mean()), float(ys.mean()))
    best = None
    for back in 'NSEW':
        it = dict(type=typ, cx=float(xs.mean()), cy=float(ys.mean()), w=sw, d=sd, ang=ANG[back], room=rid, src='assumed')
        # arranca pegado a ese muro, en línea con el centro del espacio
        if back == 'N':
            it['cy'] = float(ys.min()) + ctx.px(sd) / 2
        elif back == 'S':
            it['cy'] = float(ys.max()) - ctx.px(sd) / 2
        elif back == 'W':
            it['cx'] = float(xs.min()) + ctx.px(sd) / 2
        else:
            it['cx'] = float(xs.max()) - ctx.px(sd) / 2
        along = target[0] if back in 'NS' else target[1]
        if not _seat(ctx, it, back, rl, placed, zones, along=along):
            continue
        r_ = _rect(ctx, it)
        if sum(_inter(r_, _rect(ctx, o)) for o in placed) > 0.02 * _area(r_) or \
                sum(_inter(r_, z) for z in zones) > 0.25 * _area(r_):
            continue
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
    return out
