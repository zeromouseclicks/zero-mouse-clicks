"""Cortes arquitectónicos por reglas a partir del modelo vectorial (en metros).

Supuestos (se escriben en la lámina): altura libre 2.45 m, placa 0.15 m (entrepiso 2.60 m), puertas 2.10 m,
ventanas con antepecho 1.00 m (1.50 m en baños) y dintel 2.10 m, mesón 0.90 m, gabinete alto 1.50–2.20 m,
campana 1.55–2.20 m, nevera 1.80 m, closet a altura completa, sanitario 0.40/0.78 m, lavamanos 0.85 m,
espejo 1.20–2.00 m, ducha 2.00 m, enchape de baño hasta 2.10 m.
"""
from reportlab.lib.units import mm

CLEAR, SLAB, DOOR_H, SILL, SILL_BATH, HEAD = 2.45, 0.15, 2.10, 1.00, 1.50, 2.10


def _room_bbox(P, rid):
    poly = P.contours.get(rid)
    if not poly:
        return None
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def pick_cuts(P):
    """Corte A por la cocina (mirando hacia los mesones) y corte B por el baño mayor."""
    cuts = []
    rooms = {r['id']: r for r in P.rooms}
    counters = [o for o in P.objs if o['kind'] == 'casework' and o.get('casework') in ('counter', 'island / bar')
                and min(o['X1'] - o['X0'], o['Y1'] - o['Y0']) <= 1.3]   # >1,3 m de fondo: piso mal leído
    kitchen_ids = [r['id'] for r in P.rooms if r.get('type') == 'kitchen' or r.get('open_kitchen')]
    pool = [o for o in counters if o['room'] in kitchen_ids] or counters
    if pool:
        kfx = [f for f in P.objs if f.get('fixture') in ('cooktop', 'sink', 'fridge')]

        def _score(o):
            # preferir el mesón adosado al muro con estufa/lavaplatos; la isla solo si no hay otro
            L = max(o['X1'] - o['X0'], o['Y1'] - o['Y0'])
            on = sum(1 for f in kfx if o['X0'] - 0.3 <= (f['X0'] + f['X1']) / 2 <= o['X1'] + 0.3
                     and o['Y0'] - 0.3 <= (f['Y0'] + f['Y1']) / 2 <= o['Y1'] + 0.3)
            return L + 2.0 * min(on, 2) - (2.5 if o.get('casework') == 'island / bar' else 0)
        o = max(pool, key=_score)
        if (o['X1'] - o['X0']) >= (o['Y1'] - o['Y0']):  # mesón horizontal: corte paralelo, delante
            rb = _room_bbox(P, o['room']) or (0, 0, P.W, P.H)
            cy = (o['Y0'] + o['Y1']) / 2
            # mirar hacia el mesón desde el lado con más espacio
            if cy - rb[1] > rb[3] - cy:
                c, d = o['Y0'] - 0.75, +1
            else:
                c, d = o['Y1'] + 0.75, -1
            cuts.append(dict(axis='y', c=c, dir=d, room=o['room'], name='A', title='SECTION A-A  (KITCHEN)'))
        else:
            rb = _room_bbox(P, o['room']) or (0, 0, P.W, P.H)
            cx = (o['X0'] + o['X1']) / 2
            if cx - rb[0] > rb[2] - cx:
                c, d = o['X0'] - 0.75, +1
            else:
                c, d = o['X1'] + 0.75, -1
            cuts.append(dict(axis='x', c=c, dir=d, room=o['room'], name='A', title='SECTION A-A  (KITCHEN)'))
    baths = [r for r in P.rooms if r.get('type') == 'bath']
    if baths:
        r = max(baths, key=lambda r: r['area'])
        rb = _room_bbox(P, r['id']) or (0, 0, P.W, P.H)
        fx = [o for o in P.objs if o['room'] == r['id'] and o['kind'] in ('fixture', 'casework')]
        if rb[2] - rb[0] >= rb[3] - rb[1]:  # baño apaisado: corte horizontal por el centro
            c = (rb[1] + rb[3]) / 2
            up = sum(1 for o in fx if (o['Y0'] + o['Y1']) / 2 > c)
            d = +1 if up >= len(fx) - up else -1
            cuts.append(dict(axis='y', c=c, dir=d, room=r['id'], name='B', title='SECTION B-B  (BATHROOM)'))
        else:
            c = (rb[0] + rb[2]) / 2
            rt = sum(1 for o in fx if (o['X0'] + o['X1']) / 2 > c)
            d = +1 if rt >= len(fx) - rt else -1
            cuts.append(dict(axis='x', c=c, dir=d, room=r['id'], name='B', title='SECTION B-B  (BATHROOM)'))
    if not cuts:  # sin cocina ni baño detectados: corte longitudinal por el centro
        cuts.append(dict(axis='y', c=P.H / 2, dir=+1, room=None, name='A', title='SECTION A-A'))
    return cuts


def _frame_of(cut, P):
    """Funciones de la geometría del corte: abscisa u(x,y), rango, profundidad de vista."""
    if cut['axis'] == 'y':
        sgn = 1 if cut['dir'] > 0 else -1  # mirando +y: derecha = +x ; mirando -y: derecha = -x
        u = (lambda x, y: x) if sgn > 0 else (lambda x, y: P.W - x)
        span = P.W
    else:
        sgn = 1 if cut['dir'] > 0 else -1  # mirando +x: derecha = -y ; mirando -x: derecha = +y
        u = (lambda x, y: P.H - y) if sgn > 0 else (lambda x, y: y)
        span = P.H
    return u, span


def _in_view(cut, x0, y0, x1, y1, depth=4.5):
    c, d = cut['c'], cut['dir']
    lo, hi = (y0, y1) if cut['axis'] == 'y' else (x0, x1)
    if d > 0:
        return hi >= c and lo <= c + depth
    return lo <= c and hi >= c - depth


def _crosses(cut, x0, y0, x1, y1, tol=0.02):
    lo, hi = (y0, y1) if cut['axis'] == 'y' else (x0, x1)
    return lo - tol <= cut['c'] <= hi + tol


def _urange(cut, u, x0, y0, x1, y1):
    a, b = u(x0, y0), u(x1, y1)
    return min(a, b), max(a, b)


def _dist(cut, x0, y0, x1, y1):
    lo, hi = (y0, y1) if cut['axis'] == 'y' else (x0, x1)
    return (lo - cut['c']) if cut['dir'] > 0 else (cut['c'] - hi)


def sheet_section(c, P, meta, cut, number, frame_fn, F, FB, FRAME, TB_W, SCALES, dim_fn):
    x0, y0, x1, y1 = FRAME
    u, span = _frame_of(cut, P)
    # extensión del corte: lo que la línea atraviesa dentro de la huella
    cut_walls = [(o, a, b, cc, d) for (o, a, b, cc, d) in P.walls if _crosses(cut, a, b, cc, d)]
    # v5: los muros llegan partidos en los vanos; el tramo del vano también es muro cortado (dintel/antepecho)
    cut_walls += [(op['orient'], op['X0'], op['Y0'], op['X1'], op['Y1']) for op in P.ops
                  if _crosses(cut, op['X0'], op['Y0'], op['X1'], op['Y1'])]
    us = [x for w in cut_walls for x in _urange(cut, u, w[1], w[2], w[3], w[4])]
    full0, full1 = (min(us), max(us)) if us else (0, span)
    # ventana: el espacio protagonista (cocina o baño) más ~1.2 m a cada lado, para dibujar a escala grande
    umin, umax = full0, full1
    bb = _room_bbox(P, cut['room']) if cut.get('room') else None
    if bb:
        ra, rb_ = _urange(cut, u, bb[0], bb[1], bb[2], bb[3])
        umin, umax = max(full0, ra - 1.2), min(full1, rb_ + 1.2)
        if umax - umin < 1.0:
            umin, umax = full0, full1
    span = umax - umin
    total_h = CLEAR + 2 * SLAB
    avail_w = (x1 - TB_W) - x0 - 70 * mm
    avail_h = (y1 - y0) - 80 * mm
    scale = next((s for s in SCALES if span * 1000 / s * mm <= avail_w and total_h * 1000 / s * mm <= avail_h), SCALES[-1])
    k = 1000 / scale * mm
    ox = x0 + 35 * mm + (avail_w - span * k) / 2 - umin * k
    oy = y0 + 40 * mm + (avail_h - total_h * k) / 2 + SLAB * k

    tb = frame_fn(c, cut['title'], number, meta, f'1:{scale}  (A3)')

    def T(uu, z):
        return ox + uu * k, oy + z * k

    def rect(ua, ub, za, zb, fill=None, lw=0.3, gray=0, dash=None):
        p, q = T(ua, za), T(ub, zb)
        c.saveState(); c.setLineWidth(lw); c.setStrokeGray(gray)
        if dash:
            c.setDash(*dash)
        if fill is not None:
            c.setFillGray(fill)
        c.rect(p[0], p[1], q[0] - p[0], q[1] - p[1], stroke=1, fill=0 if fill is None else 1)
        c.restoreState()

    def line(ua, za, ub, zb, lw=0.25, gray=0, dash=None):
        p, q = T(ua, za), T(ub, zb)
        c.saveState(); c.setLineWidth(lw); c.setStrokeGray(gray)
        if dash:
            c.setDash(*dash)
        c.line(p[0], p[1], q[0], q[1]); c.restoreState()

    # todo lo dibujado queda recortado a la ventana
    c.saveState()
    pa, pb = T(umin, -SLAB - 0.05), T(umax, CLEAR + SLAB + 0.05)
    clip = c.beginPath(); clip.rect(pa[0], pa[1], pb[0] - pa[0], pb[1] - pa[1]); c.clipPath(clip, stroke=0, fill=0)

    # placas (cortadas, con trama de concreto)
    for z0 in (-SLAB, CLEAR):
        rect(umin, umax, z0, z0 + SLAB, fill=0.75, lw=0.4)
        c.saveState(); c.setLineWidth(0.1); c.setStrokeGray(0.35)
        uu = umin
        while uu < umax:
            line(uu, z0, min(uu + SLAB, umax), z0 + SLAB, lw=0.1, gray=0.35)
            uu += 0.12
        c.restoreState()
    # piso y cielo raso vistos
    line(umin, 0, umax, 0, lw=0.5); line(umin, CLEAR, umax, CLEAR, lw=0.3)

    # muros cortados (poché) y vanos cortados
    ops_cut = [op for op in P.ops if _crosses(cut, op['X0'], op['Y0'], op['X1'], op['Y1'], tol=0.08)]
    bath_ids = {r['id'] for r in P.rooms if r.get('type') == 'bath'}
    for (o, a, b, cc, d) in cut_walls:
        ua, ub = _urange(cut, u, a, b, cc, d)
        holes = []
        for op in ops_cut:
            oa, ob = _urange(cut, u, op['X0'], op['Y0'], op['X1'], op['Y1'])
            if oa < ub and ob > ua:
                holes.append((max(oa, ua), min(ob, ub), op))
        rect(ua, ub, 0, CLEAR, fill=0.2, lw=0.5)
        for ha, hb, op in holes:
            if op['kind'] == 'door':
                rect(ha, hb, 0, DOOR_H, fill=1, lw=0)
                line(ha, DOOR_H, hb, DOOR_H, lw=0.4)
            else:
                sill = SILL_BATH if cut.get('room') in bath_ids else SILL
                rect(ha, hb, sill, HEAD, fill=1, lw=0)
                mid = (ha + hb) / 2
                line(mid - 0.02, sill, mid - 0.02, HEAD, lw=0.15); line(mid + 0.02, sill, mid + 0.02, HEAD, lw=0.15)
                line(ha, sill, hb, sill, lw=0.4); line(ha, HEAD, hb, HEAD, lw=0.4)

    # elementos vistos (elevación interior), del más lejano al más cercano
    seen = [o for o in P.objs if o['kind'] in ('casework', 'fixture')
            and _in_view(cut, o['X0'], o['Y0'], o['X1'], o['Y1'])]
    # visibilidad: oculto si un muro paralelo al corte se interpone en la mayor parte de su ancho
    par = [w for w in P.walls if not _crosses(cut, w[1], w[2], w[3], w[4])]

    def visible(o):
        ua, ub = _urange(cut, u, o['X0'], o['Y0'], o['X1'], o['Y1'])
        dist_o = _dist(cut, o['X0'], o['Y0'], o['X1'], o['Y1'])
        if dist_o < -0.05:  # el objeto queda detrás del observador
            return False
        blocked = 0.0
        for w in par:
            wa, wb = _urange(cut, u, w[1], w[2], w[3], w[4])
            ov = min(ub, wb) - max(ua, wa)
            if ov <= 0:
                continue
            dw = _dist(cut, w[1], w[2], w[3], w[4])
            if -0.02 <= dw < dist_o - 0.02:
                blocked = max(blocked, ov)
        return blocked < 0.5 * (ub - ua)
    seen = [o for o in seen if visible(o)]
    seen.sort(key=lambda o: -_dist(cut, o['X0'], o['Y0'], o['X1'], o['Y1']))
    # enchape de baño en el muro del fondo
    if cut.get('room') in bath_ids:
        bb = _room_bbox(P, cut['room'])
        if bb:
            ua, ub = _urange(cut, u, bb[0], bb[1], bb[2], bb[3])
            c.saveState(); c.setLineWidth(0.08); c.setStrokeGray(0.6)
            z = 0.3
            while z <= 2.1 + 1e-6:
                line(ua, z, ub, z, lw=0.08, gray=0.6); z += 0.3
            uu = ua
            while uu <= ub:
                line(uu, 0, uu, 2.10, lw=0.08, gray=0.6); uu += 0.3
            c.restoreState()
            line(ua, 2.10, ub, 2.10, lw=0.2, gray=0.3)
    labels = []
    for o in seen:
        ua, ub = _urange(cut, u, o['X0'], o['Y0'], o['X1'], o['Y1'])
        if ub - ua < 0.05:
            continue
        kind = o.get('fixture') or o.get('casework')
        if kind == 'counter':
            rect(ua, ub, 0.10, 0.86, fill=0.97, lw=0.3)
            rect(ua, ub, 0.86, 0.90, fill=0.6, lw=0.3)
            line(ua + 0.05, 0.10, ub - 0.05, 0.10, lw=0.2)
            n = max(1, int((ub - ua) / 0.6))
            for i in range(1, n):
                line(ua + i * (ub - ua) / n, 0.10, ua + i * (ub - ua) / n, 0.86, lw=0.15)
            rect(ua, ub, 1.50, 2.20, fill=0.97, lw=0.3)  # gabinete alto
            for i in range(1, n):
                line(ua + i * (ub - ua) / n, 1.50, ua + i * (ub - ua) / n, 2.20, lw=0.15)
            labels.append(((ua + ub) / 2, 2.28, 'UPPER CABINETS'))
            labels.append(((ua + ub) / 2, 0.45, 'BASE CABINETS + COUNTERTOP'))
        elif kind == 'island / bar':
            rect(ua, ub, 0.0, 0.90, fill=0.92, lw=0.35); rect(ua, ub, 0.90, 0.94, fill=0.55, lw=0.3)
            labels.append(((ua + ub) / 2, 0.50, 'ISLAND / BAR'))
        elif kind == 'closet':
            rect(ua, ub, 0, 2.40, fill=0.96, lw=0.3)
            n = max(1, int((ub - ua) / 0.5))
            for i in range(1, n):
                line(ua + i * (ub - ua) / n, 0, ua + i * (ub - ua) / n, 2.40, lw=0.15)
            labels.append(((ua + ub) / 2, 1.20, 'CLOSET'))
        elif kind == 'vanity':
            rect(ua, ub, 0.10, 0.85, fill=0.95, lw=0.3); line(ua, 0.85, ub, 0.85, lw=0.4)
            rect(ua + 0.05, ub - 0.05, 1.20, 2.00, fill=1, lw=0.25)  # espejo
            line(ua + 0.15, 1.35, ua + 0.35, 1.85, lw=0.1, gray=0.5)
            labels.append(((ua + ub) / 2, 1.60, 'MIRROR'))
            labels.append(((ua + ub) / 2, 0.45, 'VANITY'))
        elif kind == 'toilet':
            rect(ua, ub, 0.0, 0.40, fill=1, lw=0.3); rect(ua + 0.05 * (ub - ua), ub - 0.05 * (ub - ua), 0.40, 0.78, fill=1, lw=0.3)
            labels.append(((ua + ub) / 2, 0.88, 'WC'))
        elif kind == 'lavatory':
            line((ua + ub) / 2, 0, (ua + ub) / 2, 0.75, lw=0.4)
            rect(ua, ub, 0.75, 0.85, fill=1, lw=0.3)
            rect(ua + 0.05, ub - 0.05, 1.20, 2.00, fill=1, lw=0.25)
            labels.append(((ua + ub) / 2, 1.60, 'MIRROR'))
        elif kind == 'shower':
            line(ua, 0, ua, 2.00, lw=0.2, dash=([2, 1],)); line(ub, 0, ub, 2.00, lw=0.2, dash=([2, 1],))
            line(ua, 2.00, ub, 2.00, lw=0.2, dash=([2, 1],))
            line((ua + ub) / 2, 2.00, (ua + ub) / 2, 2.15, lw=0.4); line((ua + ub) / 2 - 0.1, 2.0, (ua + ub) / 2 + 0.1, 2.0, lw=0.6)
            labels.append(((ua + ub) / 2, 1.0, 'SHOWER'))
        elif kind == 'bathtub':
            rect(ua, ub, 0.0, 0.55, fill=1, lw=0.35); labels.append(((ua + ub) / 2, 0.25, 'BATHTUB'))
        elif kind == 'fridge':
            rect(ua, ub, 0.0, 1.80, fill=1, lw=0.35); line(ua, 1.15, ub, 1.15, lw=0.2)
            labels.append(((ua + ub) / 2, 0.9, 'REF.'))
        elif kind == 'washer':
            rect(ua, ub, 0.0, 0.85, fill=1, lw=0.35)
            p = T((ua + ub) / 2, 0.45); c.circle(p[0], p[1], min(ub - ua, 0.85) * k * 0.3)
            labels.append(((ua + ub) / 2, 0.95, 'WASHER'))
        elif kind == 'cooktop':
            rect(ua, ub, 0.90, 0.93, fill=0.3, lw=0.2)
            rect(ua - 0.05, ub + 0.05, 1.55, 1.70, fill=0.9, lw=0.3)  # campana
            line((ua + ub) / 2 - 0.1, 1.70, (ua + ub) / 2 - 0.1, CLEAR, lw=0.3)
            line((ua + ub) / 2 + 0.1, 1.70, (ua + ub) / 2 + 0.1, CLEAR, lw=0.3)
            labels.append(((ua + ub) / 2, 1.45, 'HOOD'))
        elif kind == 'sink':
            line((ua + ub) / 2, 0.90, (ua + ub) / 2, 1.15, lw=0.4)
            line((ua + ub) / 2, 1.15, (ua + ub) / 2 + 0.12, 1.15, lw=0.4)
    c.setFont(F, 7 if scale <= 25 else 5.5); c.setFillGray(0.15)
    fs = 7 if scale <= 25 else 5.5
    boxes = []
    for uu, z, t in sorted(labels, key=lambda l: (l[2] != 'MIRROR', l[1])):
        p = T(uu, z); w_ = c.stringWidth(t, F, fs) / 2 + 1.5
        bx = (p[0] - w_, p[1] - 1, p[0] + w_, p[1] + fs)
        if any(bx[0] < q[2] and q[0] < bx[2] and bx[1] < q[3] and q[1] < bx[3] for q in boxes):
            continue          # se superpone con otra etiqueta: no repetir
        boxes.append(bx); c.drawCentredString(p[0], p[1], t)
    c.setFillGray(0)
    c.restoreState()
    # líneas de interrupción donde la ventana corta el edificio
    for ub_, cond in ((umin, umin > full0 + 0.01), (umax, umax < full1 - 0.01)):
        if cond:
            zs_ = [-SLAB - 0.15, 1.0, 1.15, 1.25, 1.40, CLEAR + SLAB + 0.15]
            du = [0, 0, 0.08, -0.08, 0, 0]
            c.setLineWidth(0.3)
            for (za, da), (zb, db) in zip(zip(zs_, du), zip(zs_[1:], du[1:])):
                pa, pb = T(ub_ + da, za), T(ub_ + db, zb); c.line(pa[0], pa[1], pb[0], pb[1])
    # nombres de espacios cortados (sobre la línea de piso, centrados entre muros)
    room_names = []
    for r in P.rooms:
        bb = _room_bbox(P, r['id'])
        if bb and _crosses(cut, *bb):
            ua, ub = _urange(cut, u, *bb)
            ua, ub = max(ua, umin), min(ub, umax)
            if ub - ua > 0.6:
                room_names.append(((ua + ub) / 2, r.get('name', '')))
    c.setFont(FB, 6.5)
    for uu, nm in room_names:
        p = T(uu, -SLAB - 0.32); c.drawCentredString(p[0], p[1], nm)
    # niveles
    for z, txt in ((0, 'N +0.00  FINISHED FLOOR'), (CLEAR, f'N +{CLEAR:.2f}  CEILING'),
                   (CLEAR + SLAB, f'N +{CLEAR + SLAB:.2f}  NEXT FLOOR')):
        p = T(umax, z)
        c.setLineWidth(0.3); c.line(p[0] + 3 * mm, p[1], p[0] + 9 * mm, p[1])
        c.line(p[0] + 4.5 * mm, p[1], p[0] + 6 * mm, p[1] + 1.5 * mm); c.line(p[0] + 6 * mm, p[1] + 1.5 * mm, p[0] + 7.5 * mm, p[1])
        c.setFont(F, 6); c.drawString(p[0] + 10 * mm, p[1] - 1, txt)
    # cotas verticales a la izquierda
    xl = T(umin, 0)[0] - 8 * mm
    zs = [0.0, 0.90, 2.10, CLEAR]
    for a, b in zip(zs[:-1], zs[1:]):
        dim_fn(c, (T(umin, a)[0], T(umin, a)[1]), (T(umin, b)[0], T(umin, b)[1]), xl, f'{b - a:.2f}', False, k)
    dim_fn(c, (T(umin, -SLAB)[0], T(umin, -SLAB)[1]), (T(umin, CLEAR + SLAB)[0], T(umin, CLEAR + SLAB)[1]),
           xl - 7 * mm, f'{CLEAR + 2 * SLAB:.2f}', False, k)
    # cota horizontal total del corte
    dim_fn(c, T(umin, -SLAB), T(umax, -SLAB), T(0, -SLAB - 0.75)[1], f'{umax - umin:.2f}', True, k)
    # título de vista y supuestos
    sx, sy = x0 + 12 * mm, y0 + 10 * mm
    c.setFont(FB, 10); c.drawString(sx, sy + 6 * mm, f"{cut['title']}  —  1:{scale}")
    c.setFont(F, 7)
    c.drawString(sx, sy + 2.8 * mm, f'Assumed: clear height {CLEAR:.2f} m, slab {SLAB:.2f} m (floor-to-floor {CLEAR + SLAB:.2f} m), '
                                    f'doors {DOOR_H:.2f} m, window sill {SILL:.2f} m ({SILL_BATH:.2f} m in bathrooms), counter 0.90 m.')
    c.drawString(sx, sy - 0.5 * mm, 'Schematic section derived from the plan; heights are standard assumptions — verify with the developer.')
    c.showPage()
    return scale


def draw_cut_marks(c, P, T, cuts, F, FB):
    """Línea de corte en la planta con flechas de dirección de vista."""
    for cut in cuts:
        c.saveState(); c.setLineWidth(0.5); c.setDash([8, 2, 1, 2], 0); c.setStrokeColorRGB(0.1, 0.1, 0.1)
        if cut['axis'] == 'y':
            a, b = T(-0.9, cut['c']), T(P.W + 0.9, cut['c'])
        else:
            a, b = T(cut['c'], -0.9), T(cut['c'], P.H + 0.9)
        c.line(a[0], a[1], b[0], b[1]); c.setDash()
        for p in (a, b):
            if cut['axis'] == 'y':
                dx, dy = 0, cut['dir'] * 4 * mm
            else:
                dx, dy = cut['dir'] * 4 * mm, 0
            c.setLineWidth(0.6); c.line(p[0], p[1], p[0] + dx, p[1] + dy)
            c.line(p[0] + dx, p[1] + dy, p[0] + dx * 0.6 + dy * 0.25, p[1] + dy * 0.6 + dx * 0.25)
            c.line(p[0] + dx, p[1] + dy, p[0] + dx * 0.6 - dy * 0.25, p[1] + dy * 0.6 - dx * 0.25)
            c.setFillGray(1); c.circle(p[0] - (2.5 * mm if cut['axis'] == 'x' else 0), p[1] - (2.5 * mm if cut['axis'] == 'y' else 0) * 0, 2.6 * mm, fill=1)
            c.setFillGray(0); c.setFont(FB, 7)
            c.drawCentredString(p[0] - (2.5 * mm if cut['axis'] == 'x' else 0), p[1] - 1.2 * mm, cut['name'])
        c.restoreState()
