"""Vectorizador v5 (solo reglas). Cambios frente a vectorize.py:
- los muros salen solo de la clase muro (ya filtrada por color); puertas y ventanas NO se convierten en muro
- los vanos son huecos entre muros (extremo de muro -> siguiente muro en su misma línea); la clase
  puerta/ventana del modelo solo sirve de evidencia para decidir qué es cada hueco
- espacios exteriores (patio, jardín, balcón con césped) por color; no cuentan para la escala
"""
import cv2
import numpy as np
from .vectorize import BG, SP, WL, DR, WN, main_envelope, footprint, wall_thickness, bars, snap, _cc_boxes


def _walls(mask, t):
    wl = (mask == WL).astype(np.uint8)
    L = max(5, int(round(t * 2.5)))
    hor = cv2.morphologyEx(wl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(wl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    walls = []
    for (img, o) in ((hor, 'h'), (ver.T, 'v')):
        for (a, b, c, d) in bars(img, t, L):
            walls.append([o, a, b, c, d] if o == 'h' else [o, b, a, d, c])
    cover = np.zeros_like(wl)
    for o, a, b, c, d in walls:
        cover[b:d, a:c] = 1
    rest = ((wl > 0) & (cover == 0)).astype(np.uint8)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((max(2, int(t * 0.6)),) * 2, np.uint8))
    for (x, y, w, h) in _cc_boxes(rest, int(t * t * 1.5))[0]:
        walls.append(['h' if w >= h else 'v', x, y, x + w, y + h])
    # muros demasiado delgados frente al grosor típico: líneas de muebles, no muros
    walls = [w for w in walls if min(w[3] - w[1], w[4] - w[2]) >= 0.45 * t]
    tol = max(2, int(t * 0.6))
    cy = [(b + d) / 2 for o, a, b, c, d in walls if o == 'h']
    cx = [(a + c) / 2 for o, a, b, c, d in walls if o == 'v']
    ends_x = [a for o, a, b, c, d in walls if o == 'h'] + [c for o, a, b, c, d in walls if o == 'h'] + [int(round(v)) for v in cx]
    ends_y = [b for o, a, b, c, d in walls if o == 'v'] + [d for o, a, b, c, d in walls if o == 'v'] + [int(round(v)) for v in cy]
    mx = snap([int(round(v)) for v in ends_x], tol) if ends_x else {}
    my = snap([int(round(v)) for v in ends_y], tol) if ends_y else {}
    out = []
    for o, a, b, c, d in walls:
        if o == 'h':
            th = max(2, d - b); y = my.get(int(round((b + d) / 2)), int(round((b + d) / 2)))
            out.append([o, mx.get(a, a), y - th // 2, mx.get(c, c), y - th // 2 + th])
        else:
            th = max(2, c - a); x = mx.get(int(round((a + c) / 2)), int(round((a + c) / 2)))
            out.append([o, x - th // 2, my.get(b, b), x - th // 2 + th, my.get(d, d)])
    return [w for w in out if w[3] > w[1] and w[4] > w[2]]


def _raster(walls, shape):
    W = np.zeros(shape, np.uint8)
    for o, a, b, c, d in walls:
        W[max(0, b):d, max(0, a):c] = 1
    return W


def _gap_candidates(walls, W, t, s0):
    """Huecos entre un extremo de muro y el siguiente muro sobre su misma línea."""
    h, w = W.shape
    gmax = int(8.0 / s0)
    found = []
    for o, a, b, c, d in walls:
        if o == 'h':
            if (c - a) < 1.2 * (d - b):
                continue
            band = slice(max(0, b), d)
            for start, step in ((c, 1), (a - 1, -1)):
                g = None
                k0 = 0                     # saltar la esquina: un muro perpendicular que sobresale un poco
                while k0 < 1.5 * t and 0 <= start < w and W[band, start].mean() > 0.34:
                    start += step; k0 += 1
                for i in range(gmax):
                    x = start + i * step
                    if x < 0 or x >= w:
                        break
                    if W[band, x].mean() > 0.34:
                        g = i; break
                if g is None or g < 3:
                    continue
                x0, x1 = (start, start + g) if step > 0 else (start - g + 1, start + 1)
                found.append(dict(orient='h', x0=x0, y0=b, x1=x1, y1=d))
        else:
            if (d - b) < 1.2 * (c - a):
                continue
            band = slice(max(0, a), c)
            for start, step in ((d, 1), (b - 1, -1)):
                g = None
                k0 = 0
                while k0 < 1.5 * t and 0 <= start < h and W[start, band].mean() > 0.34:
                    start += step; k0 += 1
                for i in range(gmax):
                    y = start + i * step
                    if y < 0 or y >= h:
                        break
                    if W[y, band].mean() > 0.34:
                        g = i; break
                if g is None or g < 3:
                    continue
                y0, y1 = (start, start + g) if step > 0 else (start - g + 1, start + 1)
                found.append(dict(orient='v', x0=a, y0=y0, x1=c, y1=y1))
    # sin duplicados (el mismo hueco visto desde sus dos extremos)
    uniq = []
    for gp in sorted(found, key=lambda q: (q['x1'] - q['x0']) * (q['y1'] - q['y0'])):
        dup = False
        for u in uniq:
            ix = min(gp['x1'], u['x1']) - max(gp['x0'], u['x0']); iy = min(gp['y1'], u['y1']) - max(gp['y0'], u['y0'])
            la = (gp['x1'] - gp['x0']) if gp['orient'] == 'h' else (gp['y1'] - gp['y0'])
            lb = (u['x1'] - u['x0']) if u['orient'] == 'h' else (u['y1'] - u['y0'])
            if ix > 0 and iy > 0 and u['orient'] == gp['orient'] and min(la, lb) >= 0.7 * max(la, lb):
                dup = True; break
        if not dup:
            uniq.append(gp)
    return uniq


def _classify_gaps(uniq, mask, inside, t, s0):
    """inside = dentro de la huella y no exterior (césped). Un hueco que da al exterior siempre se cierra."""
    ops = []
    dr = (mask == DR).astype(np.uint8); wn = (mask == WN).astype(np.uint8)
    # ¿este folleto marca las puertas? Si casi todos los huecos de tamaño puerta traen evidencia del modelo,
    # un hueco sin evidencia es un paso abierto; si casi ninguno la trae, todo hueco de tamaño puerta es puerta.
    ev = []
    for gp in uniq:
        ln = (gp['x1'] - gp['x0']) if gp['orient'] == 'h' else (gp['y1'] - gp['y0'])
        if 0.55 <= ln * s0 <= 1.1:
            e = max(1, int(round(t * 0.5)))
            st = dr[max(0, gp['y0'] - e):gp['y1'] + e, gp['x0']:gp['x1']] if gp['orient'] == 'h' else dr[gp['y0']:gp['y1'], max(0, gp['x0'] - e):gp['x1'] + e]
            ev.append(st.size > 0 and st.mean() >= 0.2)
    marked = sum(ev) >= 3 and sum(ev) >= 0.6 * len(ev)
    for gp in uniq:
        x0, y0, x1, y1 = gp['x0'], gp['y0'], gp['x1'], gp['y1']
        ln = (x1 - x0) if gp['orient'] == 'h' else (y1 - y0)
        th = (y1 - y0) if gp['orient'] == 'h' else (x1 - x0)
        m = ln * s0
        e = max(1, int(round(t * 0.5)))
        if gp['orient'] == 'h':
            strip = (slice(max(0, y0 - e), y1 + e), slice(x0, x1))
            swing = (slice(max(0, y0 - ln), y1 + ln), slice(max(0, x0 - e), x1 + e))
            side_a = inside[max(0, y0 - int(2 * t)):max(1, y0 - e), x0:x1]; side_b = inside[y1 + e:y1 + int(2 * t), x0:x1]
        else:
            strip = (slice(y0, y1), slice(max(0, x0 - e), x1 + e))
            swing = (slice(max(0, y0 - e), y1 + e), slice(max(0, x0 - ln), x1 + ln))
            side_a = inside[y0:y1, max(0, x0 - int(2 * t)):max(1, x0 - e)]; side_b = inside[y0:y1, x1 + e:x1 + int(2 * t)]
        wn_f = float(wn[strip].mean()) if wn[strip].size else 0.0
        dr_f = float(dr[strip].mean()) if dr[strip].size else 0.0
        dr_sw = float(dr[swing].sum()) / max(1.0, ln * max(th, t))
        oa = side_a.size == 0 or side_a.mean() < 0.5; ob = side_b.size == 0 or side_b.mean() < 0.5
        if oa and ob:
            continue                          # hueco con exterior a ambos lados: no es un vano
        ext = oa or ob
        kind = None; weak = False
        if wn_f >= 0.30 and wn_f >= dr_f and 0.4 <= m <= 4.5:
            kind = 'window'
        elif 0.5 <= m <= 1.35 and dr_f >= 0.2:
            kind = 'door'
        elif 1.35 < m <= 2.6 and dr_f >= 0.45:
            kind = 'door'                      # puerta doble / corrediza
        elif not ext and 0.55 <= m <= 1.1:
            kind = 'door'; weak = True         # hueco de tamaño puerta sin evidencia del modelo
        elif ext and 0.4 <= m <= 8.0:
            kind = 'door' if m <= 1.1 and dr_f > 0.08 else 'window'   # frente acristalado / ventanal
        elif m < 0.4:
            kind = 'wall'                      # hueco mínimo entre dos muros: muro mal leído, se cierra
        if kind:
            ops.append(dict(kind=kind, orient=gp['orient'], x0=x0, y0=y0, x1=x1, y1=y1, weak=weak, marked=bool(marked),
                            ev=dict(wn=round(wn_f, 2), dr=round(dr_f, 2), sw=round(dr_sw, 2), ext=bool(ext), m=round(m, 2))))
    return ops


def exterior_fraction(image, sel):
    """Fracción de píxeles verdes (césped, plantas) dentro de una selección."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    px = hsv[sel]
    if len(px) == 0:
        return 0.0
    g = (px[:, 0] >= 30) & (px[:, 0] <= 95) & (px[:, 1] >= 55) & (px[:, 2] >= 35)
    return float(g.mean())


def green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    g = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 55) & (hsv[..., 2] >= 35)
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0


def _footprint(W, cands, mask, t):
    """Huella: lo encerrado por muros + huecos candidatos + vanos del modelo."""
    env = (W > 0) | np.isin(mask, (DR, WN))
    env = env.astype(np.uint8)
    for g in cands:
        env[g['y0']:g['y1'], g['x0']:g['x1']] = 1
    k = max(3, int(round(t * 1.6)) | 1)
    closed = cv2.morphologyEx(env, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    h, w = closed.shape
    big = np.zeros((h + 2, w + 2), np.uint8); big[1:-1, 1:-1] = closed
    pad = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(big, pad, (0, 0), 2)
    inside = (big[1:-1, 1:-1] != 2).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inside, 8)
    if n <= 1:
        return inside.astype(bool)
    return lab == (1 + int(np.argmax(st[1:, 4])))


def vectorize5(mask, area_m2=None, image=None):
    mask = main_envelope(mask)
    t = wall_thickness(mask)
    walls = _walls(mask, t)
    W = _raster(walls, mask.shape)
    s0 = float(np.sqrt(area_m2 / (0.8 * max(1, np.count_nonzero(footprint(mask)))))) if area_m2 else 0.02
    cands = _gap_candidates(walls, W, t, s0)
    fp = _footprint(W, cands, mask, t)
    green = green_mask(image) if image is not None else np.zeros(mask.shape, bool)
    s0 = float(np.sqrt(area_m2 / (0.88 * max(1, (fp & ~green).sum())))) if area_m2 else 0.02
    cands = [g for g in _gap_candidates(walls, W, t, s0)]
    mask[~fp] = BG
    openings = _classify_gaps(cands, mask, fp & ~green, t, s0)
    for g in [o for o in openings if o['kind'] == 'wall']:
        walls.append([g['orient'], g['x0'], g['y0'], g['x1'], g['y1']])
        W[g['y0']:g['y1'], g['x0']:g['x1']] = 1
    openings = [o for o in openings if o['kind'] != 'wall']
    H_, W_ = mask.shape
    dr = (mask == DR).astype(np.float32)

    def segment(openings):
        closure = W.copy()
        for op in openings:
            closure[op['y0']:op['y1'], op['x0']:op['x1']] = 1
        closure = cv2.dilate(closure, np.ones((3, 3), np.uint8))
        inner = fp & ~(closure > 0)
        inner = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(inner, 4)
        min_room = (t * 6) ** 2
        rooms = []
        for i in range(1, n):
            if st[i, 4] < min_room:
                continue
            r = dict(id=len(rooms) + 1, label=i, area_px=int(st[i, 4]), cx=float(cen[i, 0]), cy=float(cen[i, 1]),
                     bbox=[int(q) for q in st[i, :4]])
            r['green'] = float(green[lab == i].mean())
            r['exterior'] = r['green'] >= 0.5
            rooms.append(r)
        return lab, rooms

    def side_room(op, sgn, lab, by):
        """Espacio (id) al lado sgn (-1/+1) del vano, o None si da afuera."""
        e = max(3, int(2.0 * t)); q = 2
        if op['orient'] == 'h':
            ys = slice(max(0, op['y0'] - e), max(0, op['y0'] - q)) if sgn < 0 else slice(min(H_, op['y1'] + q), min(H_, op['y1'] + e))
            sub = lab[ys, op['x0'] + 1:max(op['x0'] + 2, op['x1'] - 1)]
        else:
            xs = slice(max(0, op['x0'] - e), max(0, op['x0'] - q)) if sgn < 0 else slice(min(W_, op['x1'] + q), min(W_, op['x1'] + e))
            sub = lab[op['y0'] + 1:max(op['y0'] + 2, op['y1'] - 1), xs]
        sub = sub.ravel(); sub = sub[sub > 0]
        if sub.size == 0:
            return None
        lb = int(np.bincount(sub).argmax())
        return by[lb]['id'] if lb in by else None

    for _ in range(3):
        lab, rooms = segment(openings)
        by = {r['label']: r for r in rooms}
        byid = {r['id']: r for r in rooms}
        for op in openings:
            op['rooms'] = [side_room(op, -1, lab, by), side_room(op, +1, lab, by)]
        # vestíbulo diminuto creado por un hueco sin evidencia (boca de un pasillo corto): se reabre.
        # También sobra un «vano» que tiene el mismo espacio a ambos lados.
        drop = []
        for op in openings:
            ra, rb = op['rooms']
            if op['kind'] == 'door' and ra is not None and ra == rb:
                drop.append(op); continue
            if op['kind'] != 'door' or ra is None or rb is None:
                continue
            for rid, oid in ((ra, rb), (rb, ra)):
                r = byid.get(rid)
                if r is None or r['area_px'] * s0 * s0 >= 2.5:
                    continue
                mine = [o for o in openings if o['kind'] == 'door' and rid in o['rooms'] and None not in o['rooms']]
                if len(mine) < 3:
                    continue
                # vestíbulo de paso con tres o más puertas: se une al vecino más grande (la zona social)
                nb = max((o['rooms'][0] if o['rooms'][1] == rid else o['rooms'][1] for o in mine),
                         key=lambda i: byid[i]['area_px'] if i in byid else 0)
                if oid == nb:
                    drop.append(op); break
        if not drop:
            break
        openings = [o for o in openings if not any(o is q for q in drop)]
    for op in openings:
        if op.get('weak') and op.get('marked'):
            op['style'] = 'opening'            # el folleto marca sus puertas y aquí no hay: paso sin puerta

    outd = []
    for op in openings:
        ra, rb = op['rooms']
        ln = (op['x1'] - op['x0']) if op['orient'] == 'h' else (op['y1'] - op['y0'])
        if op['kind'] == 'door':
            # hoja: hacia el lado donde el modelo vio la hoja/arco; bisagra en el extremo más cargado
            if op['orient'] == 'h':
                za = dr[max(0, op['y0'] - ln):op['y0'], op['x0']:op['x1']]; zb = dr[op['y1']:op['y1'] + ln, op['x0']:op['x1']]
                z = za if za.sum() >= zb.sum() else zb
                col = z.sum(0) if z.size else np.zeros(1)
            else:
                za = dr[op['y0']:op['y1'], max(0, op['x0'] - ln):op['x0']]; zb = dr[op['y0']:op['y1'], op['x1']:op['x1'] + ln]
                z = za if za.sum() >= zb.sum() else zb
                col = z.sum(1) if z.size else np.zeros(1)
            op['side'] = -1 if za.sum() >= zb.sum() else 1
            half = len(col) // 2
            op['hinge'] = 0 if col[:half].sum() >= col[half:].sum() else 1
            if ra is None or rb is None:
                outd.append(op)
    if outd:
        keep = max(outd, key=lambda o: o['ev']['dr'] * min(o['ev']['m'], 1.0))
        for o in outd:
            if o is not keep:
                o['kind'] = 'window'
        keep['entry'] = True
    # muros: solo lo que bordea espacios o vanos (colas de muro hacia escaleras o zonas comunes, fuera)
    near = np.zeros(mask.shape, np.uint8)
    for r in rooms:
        near[lab == r['label']] = 1
    for op in openings:
        near[op['y0']:op['y1'], op['x0']:op['x1']] = 1
    kk = 2 * int(round(2.2 * t)) + 1
    near = cv2.dilate(near, np.ones((kk, kk), np.uint8))
    trimmed = []
    for o, a, b, c, d in walls:
        sub = near[b:d, a:c]
        if not sub.any():
            continue
        if o == 'h':
            cols = np.flatnonzero(sub.any(0)); a, c = a + int(cols[0]), a + int(cols[-1]) + 1
        else:
            rws = np.flatnonzero(sub.any(1)); b, d = b + int(rws[0]), b + int(rws[-1]) + 1
        trimmed.append([o, a, b, c, d])
    walls = trimmed
    inner_px = sum(r['area_px'] for r in rooms if not r['exterior']) or int(fp.sum())
    m_per_px = float(np.sqrt(area_m2 / inner_px)) if area_m2 and inner_px else None
    bbox = [min(w[1] for w in walls), min(w[2] for w in walls), max(w[3] for w in walls), max(w[4] for w in walls)] if walls else [0, 0, mask.shape[1], mask.shape[0]]
    return dict(thickness_px=t, m_per_px=m_per_px, area_m2=area_m2, bbox=bbox,
                walls=walls, openings=openings, rooms=rooms, shape=list(mask.shape),
                footprint=fp, room_labels=lab, scale_from='declared area', inner_px=inner_px)
