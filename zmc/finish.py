"""Versión acabada del apartamento, solo con reglas (sin IA):
- objetos fijos (mesones/closets/barras, aparatos) y muebles sueltos a partir de la máscara de 8 clases
- tipo de cada espacio según su contenido y tamaño
- acabado de piso de cada espacio según la textura de la imagen
- propuesta ESQUEMÁTICA de tomas, interruptores y puntos de luz
Todo en píxeles de la máscara; drawing.py lo pasa a metros.
"""
import cv2
import numpy as np

from .vectorize import WL, DR, WN, MS, AP, MU


def _room_of(lab, room_by_label, x, y, w, h):
    sub = lab[y:y + h, x:x + w].ravel()
    sub = sub[sub > 0]
    if sub.size == 0:  # objeto pegado al muro: mirar un poco alrededor
        sub = lab[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4].ravel()
        sub = sub[sub > 0]
    if sub.size == 0:
        return None
    lb = np.bincount(sub).argmax()
    return room_by_label.get(int(lb))


def extract_objects(mask, model):
    s = model['m_per_px']
    fp = model['footprint']
    lab = model['room_labels']
    room_by_label = {r['label']: r['id'] for r in model['rooms']}
    min_px = max(6, int((0.12 / s) ** 2))
    objs = []
    for cls, kind in ((MS, 'casework'), (AP, 'fixture'), (MU, 'furniture')):
        b = ((mask == cls) & fp).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, l, st, _ = cv2.connectedComponentsWithStats(b, 8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < min_px:
                continue
            fill = a / float(w * h)
            lo, hi = sorted((w * s, h * s))
            if kind == 'fixture' and (lo < 0.22 or a * s * s < 0.06):
                continue      # cojines, sillas, reflejos
            if kind == 'casework' and (lo < 0.25 or hi < 0.45):
                continue
            objs.append(dict(kind=kind, x0=int(x), y0=int(y), x1=int(x + w), y1=int(y + h),
                             w_m=w * s, h_m=h * s, fill=fill,
                             room=_room_of(lab, room_by_label, x, y, w, h)))
    ext = {r['id'] for r in model['rooms'] if r.get('exterior')}
    return [o for o in objs if o['room'] not in ext]


ROOM_NAMES = {'patio': 'PATIO', 'living': 'LIVING / DINING', 'kitchen': 'KITCHEN', 'bed': 'BEDROOM', 'bath': 'BATHROOM',
              'laundry': 'LAUNDRY', 'closet': 'CLOSET', 'hall': 'HALL', 'study': 'STUDY', 'balcony': 'BALCONY'}


def _dims(o):
    return sorted((o['w_m'], o['h_m']))


def classify(model, objs, image=None):
    """Tipo de espacio y tipo de aparato, por reglas de contenido y tamaño.
    v4.1: candidatos con puntaje y cupos por tamaño del apartamento (máx. 1 cocina, 1 ropas,
    baños ~ 1 por cada 35 m²), para que sillas o cojines confundidos con aparatos no creen baños."""
    s2 = model['m_per_px'] ** 2
    rooms = {r['id']: dict(r, area=r['area_px'] * s2, kinds=[]) for r in model['rooms']}
    for o in objs:
        if o['room'] in rooms:
            rooms[o['room']]['kinds'].append(o)
    big = max(rooms.values(), key=lambda r: r['area'])['id'] if rooms else None
    total = sum(r['area'] for r in rooms.values()) or 1.0
    n_bath = int(np.clip(np.ceil(total / 30.0), 1, 4))
    for r in rooms.values():
        fx = [o for o in r['kinds'] if o['kind'] == 'fixture']
        cw = [o for o in r['kinds'] if o['kind'] == 'casework']
        fu = [o for o in r['kinds'] if o['kind'] == 'furniture']
        # cama: rectángulo lleno de al menos 0,8 x 1,45 m (camas sencillas de folleto salen algo pequeñas)
        r['bed'] = any(_dims(o)[0] >= 0.8 and _dims(o)[1] >= 1.45 and o['fill'] >= 0.75 for o in fu)
        # mueble grande tipo cama (aunque las almohadas le quiten relleno): descarta cocina y baño
        r['bedlike'] = r['bed'] or any(_dims(o)[0] >= 0.85 and o['w_m'] * o['h_m'] >= 1.6 and o['fill'] >= 0.55 for o in fu)
        # sanitario/lavamanos: ovalado (llena < 90 % de su caja); lavadora: cuadrada y llena
        r['toilets'] = sum(1 for o in fx if 0.28 <= _dims(o)[0] and _dims(o)[1] <= 0.9
                           and o['w_m'] * o['h_m'] <= 0.6)
        r['washers'] = sum(1 for o in fx if 0.5 <= _dims(o)[0] and _dims(o)[1] <= 0.8 and o['fill'] >= 0.9)
        r['wet'] = sum(1 for o in fx if _dims(o)[0] >= 0.7)          # ducha / tina
        r['cw_len'] = sum(_dims(o)[1] for o in cw if _dims(o)[0] >= 0.35)
        r['nfx'] = len(fx)
        x, y, w, h = r['bbox']
        r['aspect'] = max(w, h) / max(1, min(w, h))
        r['type'] = None
    # v5: lo que dice el folleto manda (rótulos leídos por OCR) y los espacios con césped son exteriores
    for r in rooms.values():
        ts = [t['type'] for t in r.get('texts', [])]
        if r.get('exterior') or (ts and all(t == 'patio' for t in ts)):
            r['type'] = 'patio'; r['exterior'] = True
        elif 'living' in ts or ('kitchen' in ts and r['id'] == big and 'bed' not in ts):
            r['type'] = 'living'
            if 'kitchen' in ts:
                r['open_kitchen'] = True
        elif ts:
            r['type'] = max(set(ts), key=ts.count)
        r['forced'] = r['type'] is not None
    has = lambda t: any(r['type'] == t for r in rooms.values())
    labelled = sum(1 for r in rooms.values() if r['forced'] and r['type'] != 'patio')
    n_bath = max(0, n_bath - sum(1 for r in rooms.values() if r['type'] == 'bath'))
    inter = [r for r in rooms.values() if r['type'] is None]
    if not has('living') and inter:
        big = max(inter, key=lambda r: r['area'])['id']
        rooms[big]['type'] = 'living'
    free = lambda: [r for r in rooms.values() if r['type'] is None]
    # baños
    for r in rooms.values():
        r['bigfu'] = any(o['kind'] == 'furniture' and o['w_m'] * o['h_m'] >= 0.5 and o['fill'] >= 0.75
                         for o in r['kinds'])
    cand = [r for r in free() if 1.6 <= r['area'] <= 9.5 and not r['bed'] and not r['bigfu']
            and (r['toilets'] or r['wet'])]
    score = lambda r: (2 * min(r['toilets'], 2) + r['wet'] + 3 * (r['area'] < 6) - 3 * (r['area'] > 7))
    cand = sorted([r for r in cand if score(r) >= 4], key=lambda r: -score(r))
    for r in cand[:n_bath]:
        r['type'] = 'bath'
    if not cand and not has('bath'):   # el modelo no vio ningún aparato sanitario: el espacio pequeño más «vacío»
        alt = [r for r in free() if 1.6 <= r['area'] <= 6 and not r['bed'] and not r['bigfu']]   # todo apartamento tiene baño
        alt.sort(key=lambda r: (r['cw_len'] > 0.5, r['area']))
        for r in alt[:1]:
            r['type'] = 'bath'
    # cocina (una; dos solo en apartamentos muy grandes)
    # una cocina cerrada es pequeña frente al apartamento; un espacio grande con closets es una alcoba
    cand = [r for r in free() if 3 <= r['area'] <= max(9.0, 0.16 * total) and not r['bedlike'] and r['cw_len'] >= 1.4
            and (r['nfx'] or r['cw_len'] >= 2.5) and r['wet'] == 0]
    cand.sort(key=lambda r: -(r['cw_len'] + 0.8 * r['nfx']))
    liv = next((r for r in rooms.values() if r['type'] == 'living'), None)
    if has('kitchen') or (liv is not None and liv.get('open_kitchen')):
        cand = []                      # el folleto ya dijo dónde está la cocina
    elif liv is not None and liv['cw_len'] >= 1.5 and (not cand or liv['cw_len'] + 0.8 * liv['nfx'] >
                                                      1.5 * (cand[0]['cw_len'] + 0.8 * cand[0]['nfx'])):
        liv['open_kitchen'] = True        # cocina abierta dentro de la sala-comedor
        cand = cand[1:2] if total > 160 else []
    for r in cand[:1 + (total > 160)]:
        r['type'] = 'kitchen'
    # ropas
    cand = [r for r in free() if r['area'] < 6 and r['nfx'] and not r['bed']] if not has('laundry') else []
    if labelled >= 3:
        cand = [r for r in cand if r['area'] < 4.5 and r['cw_len'] < 1.0]   # folleto rotulado: ropas solo si es muy claro
    cand.sort(key=lambda r: -r['nfx'])
    for r in cand[:1]:
        r['type'] = 'laundry'
    # vecinos por puerta: un espacio sin cama que reparte a varias alcobas/estudio/ropas es circulación
    lab = model['room_labels']
    by_label = {r['label']: r for r in rooms.values()}
    nb = {rid: set() for rid in rooms}
    for d in model['openings']:
        if d['kind'] != 'door':
            continue
        sub = lab[max(0, d['y0'] - 6):d['y1'] + 6, max(0, d['x0'] - 6):d['x1'] + 6]
        ids = {by_label[int(v)]['id'] for v in np.unique(sub) if int(v) in by_label}
        for i in ids:
            nb[i] |= ids - {i}
    big_free = [r for r in free() if r['area'] >= 7.5 and not r['bed'] and r['aspect'] <= 2.8]
    for r in rooms.values():
        r['deg'] = len(nb[r['id']])
    hub = max(big_free, key=lambda r: (r['deg'], -r['cw_len'])) if big_free else None
    if hub is not None and (total < 80 or hub['deg'] < (2 if total >= 100 else 3)):
        hub = None
    for r in free():
        if r['aspect'] > 2.8 and not r['bed']:
            t = 'hall'
        elif r['bed']:
            t = 'bed'
        elif r is hub:
            t = 'hall'           # estar de alcobas / vestíbulo (p. ej. con tapete redondo)
        elif r['area'] >= 7.0 or (r['area'] >= 5.0 and r['cw_len'] >= 0.9 and r['aspect'] <= 2.2):
            t = 'bed'           # alcoba pequeña con closet (vivienda compacta)
        elif r['area'] < 2.5:
            t = 'closet'
        elif r['area'] < 5 or r['aspect'] > 2.8:
            t = 'hall'
        else:
            t = 'study'
        r['type'] = t
    # numerar alcobas y baños
    cnt = {}
    for r in sorted(rooms.values(), key=lambda r: -r['area']):
        cnt[r['type']] = cnt.get(r['type'], 0) + 1
        r['n'] = cnt[r['type']]
    prim = [r for r in rooms.values() if r['type'] == 'bed' and any(t.get('primary') for t in r.get('texts', []))]
    if prim:                       # la principal va primero, las demás por tamaño
        beds = prim[:1] + sorted([r for r in rooms.values() if r['type'] == 'bed' and r is not prim[0]], key=lambda r: -r['area'])
        for i, r in enumerate(beds):
            r['n'] = i + 1
    for r in rooms.values():
        base = ROOM_NAMES[r['type']]
        own = [t['name'] for t in r.get('texts', []) if t['type'] == r['type']]
        if r['type'] in ('patio', 'closet', 'study', 'hall') and own:
            base = own[0]
        r['name'] = f"{base} {r['n']}" if cnt[r['type']] > 1 and r['type'] in ('bed', 'bath') else base
        if prim and r is prim[0]:
            r['name'] = 'PRIMARY BEDROOM'
        if r.get('open_kitchen'):
            r['name'] = 'LIVING / DINING / KITCHEN'
        zs = {}
        for t in r.get('texts', []):
            zs.setdefault(t['name'], (t['mx'], t['my']))
        r['zones'] = [(n, x, y) for n, (x, y) in zs.items()] if len(zs) >= 2 else []
    # v5: zona de cocina dentro de un espacio abierto. Solo lo que está junto a la cocina es mesón o aparato;
    # lo que queda más cerca de la sala (sofás, mesas de centro) es mueble suelto.
    s = model['m_per_px']
    extra = []
    for r in rooms.values():
        if not (r.get('open_kitchen') or r['type'] == 'kitchen'):
            continue
        inroom = [o for o in objs if o['room'] == r['id'] and o['kind'] in ('fixture', 'casework')]
        ks = [t for t in r.get('texts', []) if t['type'] == 'kitchen']
        ls = [t for t in r.get('texts', []) if t['type'] == 'living']
        if r.get('open_kitchen'):
            if ks:
                ax, ay = ks[0]['mx'], ks[0]['my']
            else:
                cws = [o for o in inroom if _dims(o)[1] >= 1.2 and _dims(o)[0] <= 0.9]
                if not cws:
                    continue
                b = max(cws, key=lambda o: _dims(o)[1]); ax, ay = (b['x0'] + b['x1']) / 2, (b['y0'] + b['y1']) / 2
            def dist(o, px, py):
                return float(np.hypot(max(o['x0'] - px, 0, px - o['x1']), max(o['y0'] - py, 0, py - o['y1']))) * s
            for o in inroom:
                dk = dist(o, ax, ay)
                dl = min([dist(o, t['mx'], t['my']) for t in ls] or [1e9])
                if dk > 2.6 or dl < dk:
                    o['kind'] = 'furniture'
            r['kitchen_anchor'] = (ax, ay)
        # mesón que el modelo leyó como un solo «aparato» alargado (mesón + estufa + lavaplatos)
        for o in [o for o in objs if o['room'] == r['id'] and o['kind'] == 'fixture']:
            a, b = _dims(o)
            if b >= 1.2 and 0.4 <= a <= 0.9:
                o['kind'] = 'casework'; o['composite'] = True
                if image is not None:
                    extra += _counter_fixtures(image, o, s)
        if image is not None:
            extra += _white_appliances(image, model, r, objs + extra, s)
    # tipo de aparato según tamaño y espacio
    for o in objs:
        if o['kind'] != 'fixture':
            continue
        a, b = sorted((o['w_m'], o['h_m']))
        rt = rooms.get(o['room'], {}).get('type', 'living')
        cx_, cy_ = (o['x0'] + o['x1']) / 2, (o['y0'] + o['y1']) / 2
        tol = 0.15 / model['m_per_px']
        kit = rt == 'kitchen' or rooms.get(o['room'], {}).get('open_kitchen')
        on_counter = (kit or rt == 'bath') and any(c['kind'] == 'casework' and c['x0'] - tol <= cx_ <= c['x1'] + tol
                         and c['y0'] - tol <= cy_ <= c['y1'] + tol for c in objs)
        if b >= 1.3 and a >= 0.55 and rt != 'bath' and not on_counter:
            o['kind'] = 'furniture'       # «tina» fuera del baño: casi siempre un sofá o mueble
            continue
        if b >= 1.3 and a >= 0.55:
            k = 'bathtub'
        elif rt == 'bath':
            if a >= 0.7:
                k = 'shower'
            elif on_counter or o['fill'] < 0.8 and b / max(a, 1e-3) < 1.25:
                k = 'lavatory'
            else:
                k = 'toilet'
        elif on_counter:
            k = 'cooktop' if o['w_m'] * o['h_m'] > 0.2 else 'sink'
        elif rt == 'laundry':
            k = 'washer'
        elif rt == 'kitchen' or rooms.get(o['room'], {}).get('open_kitchen'):
            k = 'fridge' if a >= 0.62 else 'washer'
        else:          # aparato suelto en sala, alcoba o estudio: casi siempre un mueble mal leído
            o['kind'] = 'furniture'
            continue
        o['fixture'] = k
    # cocina: una estufa (el aparato de mesón más grande) y hasta dos lavaplatos; el resto, muebles
    for rid, r in rooms.items():
        on = sorted([o for o in objs if o['room'] == rid and o.get('fixture') in ('cooktop', 'sink')],
                    key=lambda o: -o['w_m'] * o['h_m'])
        for i, o in enumerate(on):
            if i == 0:
                o['fixture'] = 'cooktop'
            elif i <= 2:
                o['fixture'] = 'sink'
            else:
                o['kind'] = 'furniture'; o.pop('fixture')
    # cupos: una nevera y hasta dos lavadoras/secadoras por espacio
    for rid, r in rooms.items():
        for k, cap in (('fridge', 1), ('washer', 2)):
            on = sorted([o for o in objs if o['room'] == rid and o.get('fixture') == k], key=lambda o: -o['w_m'] * o['h_m'])
            for o in on[cap:]:
                o['kind'] = 'furniture'; o.pop('fixture')
    for rid, r in rooms.items():
        if r['type'] != 'bath':
            continue
        small = [o for o in objs if o['room'] == rid and o.get('fixture') in ('toilet', 'lavatory')]
        if small:
            best = max(small, key=lambda o: max(o['w_m'], o['h_m']) / max(min(o['w_m'], o['h_m']), 1e-3)
                       + (0 if any(c['kind'] == 'casework' and c['x0'] <= o['x0'] <= c['x1'] and c['y0'] <= o['y0'] <= c['y1']
                                   for c in objs) else 1))
            for o in small:
                o['fixture'] = 'toilet' if o is best else 'lavatory'
    for o in objs:
        if o['kind'] == 'casework':
            rt = rooms.get(o['room'], {}).get('type', 'living')
            touches_wall = True  # aproximación: casi siempre adosado
            if rt in ('bed', 'closet', 'hall', 'study') and not (0.35 <= min(o['w_m'], o['h_m']) <= 0.85 and max(o['w_m'], o['h_m']) >= 0.7
                                                              and _niche(model, o, rooms.get(o['room']))):
                o['kind'] = 'furniture'; continue       # cama, escritorio o mesa leídos como mueble fijo
            if (rt == 'kitchen' or rooms.get(o['room'], {}).get('open_kitchen')) and not o.get('composite') \
                    and 0.4 <= min(o['w_m'], o['h_m']) and max(o['w_m'], o['h_m']) <= 0.9:
                have = [q.get('fixture') for q in objs + extra if q['room'] == o['room']]
                if have.count('washer') < 2:
                    o['kind'] = 'fixture'; o['fixture'] = 'washer' if 'fridge' in have else 'fridge'
                    continue
            o['casework'] = 'closet' if rt in ('bed', 'closet', 'hall', 'study') else \
                ('vanity' if rt == 'bath' else ('island / bar' if min(o['w_m'], o['h_m']) >= 0.8 else 'counter'))
    objs.extend(extra)
    return rooms


def _counter_fixtures(image, o, s):
    """Estufa y lavaplatos sobre un mesón. La superficie del mesón es el tono claro dominante; la estufa
    es el tramo que más se oscurece (quemadores) y el lavaplatos el siguiente tramo distinto. Aproximado."""
    g = cv2.cvtColor(image[o['y0']:o['y1'], o['x0']:o['x1']], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.size == 0:
        return []
    vert = g.shape[0] >= g.shape[1]
    if not vert:
        g = g.T
    L = g.shape[0]
    k = max(3, int(0.2 / s))
    rm = np.convolve(g.mean(1), np.ones(k) / k, mode='same')
    diff = np.percentile(rm, 85) - rm
    if diff.max() < 18:
        return []
    def box(center, length, kind):
        half = int(length / s / 2)
        p0 = int(np.clip(center - half, 0, max(0, L - 2 * half))); p1 = min(L, p0 + 2 * half)
        m = max(1, int(0.05 / s))
        if vert:
            x0, x1, y0, y1 = o['x0'] + m, o['x1'] - m, o['y0'] + p0, o['y0'] + p1
        else:
            x0, x1, y0, y1 = o['x0'] + p0, o['x0'] + p1, o['y0'] + m, o['y1'] - m
        return dict(kind='fixture', fixture=kind, x0=x0, y0=y0, x1=x1, y1=y1, w_m=(x1 - x0) * s, h_m=(y1 - y0) * s,
                    fill=1.0, room=o['room'], derived=True), p0, p1
    ck, c0, c1 = box(int(np.argmax(diff)), 0.55, 'cooktop')
    out = [ck]
    other = diff > 18
    other[max(0, c0 - k):c1 + k] = False
    runs, st = [], None
    for i, v in enumerate(list(other) + [False]):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((st, i)); st = None
    runs = [r for r in runs if (r[1] - r[0]) * s >= 0.25]
    if runs:
        a_, b_ = max(runs, key=lambda r: r[1] - r[0])
        out.append(box((a_ + b_) // 2, min(0.6, (b_ - a_) * s), 'sink')[0])
    return out


def _niche(model, o, room):
    """Un closet va de muro a muro (nicho) o cubre casi todo un lado del espacio; un escritorio o una cama no."""
    t = max(3, int(1.2 * model['thickness_px']))
    W = np.zeros(model['shape'][:2], np.uint8)
    for _, a, b, c, d in model['walls']:
        W[max(0, b):d, max(0, a):c] = 1
    if (o['x1'] - o['x0']) >= (o['y1'] - o['y0']):
        e0 = W[o['y0']:o['y1'], max(0, o['x0'] - t):o['x0']]; e1 = W[o['y0']:o['y1'], o['x1']:o['x1'] + t]
        span = (o['x1'] - o['x0']) / max(1, room['bbox'][2]) if room else 0
    else:
        e0 = W[max(0, o['y0'] - t):o['y0'], o['x0']:o['x1']]; e1 = W[o['y1']:o['y1'] + t, o['x0']:o['x1']]
        span = (o['y1'] - o['y0']) / max(1, room['bbox'][3]) if room else 0
    touch = int(e0.size > 0 and e0.any()) + int(e1.size > 0 and e1.any())
    return touch == 2 or span >= 0.7


def _white_appliances(image, model, r, objs, s):
    """Nevera / lavadora dibujadas como cajas blancas junto a la cocina (el modelo las confunde con fondo)."""
    lab = model['room_labels']
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] >= 228) & (hsv[..., 1] <= 28) & (lab == r['label'])
    for o in objs:
        if o['kind'] != 'furniture':
            white[o['y0']:o['y1'], o['x0']:o['x1']] = False
    white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(white, 4)
    out = []
    anc = r.get('kitchen_anchor')
    for i in range(1, n):
        x, y, w, h, a = st[i]
        lo, hi = sorted((w * s, h * s))
        if a / float(w * h) < 0.8 or not (0.45 <= lo <= 0.95):
            continue
        if anc is not None and np.hypot(x + w / 2 - anc[0], y + h / 2 - anc[1]) * s > 2.6:
            continue
        boxes = []
        if hi <= 0.95:
            boxes = [(x, y, w, h)]
        elif 1.0 <= hi <= 1.9:                      # dos aparatos pegados
            boxes = [(x, y, w, h // 2), (x, y + h // 2, w, h - h // 2)] if h > w else [(x, y, w // 2, h), (x + w // 2, y, w - w // 2, h)]
        for (bx, by, bw, bh) in boxes:
            out.append(dict(kind='fixture', x0=int(bx), y0=int(by), x1=int(bx + bw), y1=int(by + bh), w_m=bw * s, h_m=bh * s,
                            fill=1.0, room=r['id'], derived=True))
    out.sort(key=lambda o: -o['w_m'] * o['h_m'])
    for i, o in enumerate(out[:3]):
        o['fixture'] = 'fridge' if i == 0 else 'washer'
    return out[:3]


def doors(model, rooms):
    """Sentido de las puertas: alcobas, baños y ropas abren hacia adentro del espacio; el acceso abre hacia
    el interior; las puertas anchas a patio/balcón son corredizas."""
    s = model['m_per_px']
    priv = ('bed', 'bath', 'study', 'laundry', 'closet')
    for op in model['openings']:
        if op['kind'] != 'door' or 'rooms' not in op:
            continue
        ra, rb = op['rooms']
        ta = rooms[ra]['type'] if ra in rooms else None
        tb = rooms[rb]['type'] if rb in rooms else None
        w = ((op['x1'] - op['x0']) if op['orient'] == 'h' else (op['y1'] - op['y0'])) * s
        if op.get('entry'):
            op['side'] = -1 if ra is not None else 1
            op['out'] = 1 if rb is None else -1
        elif 'patio' in (ta, tb):
            op['style'] = 'sliding' if w >= 1.0 else op.get('style', 'swing')
            op['side'] = -1 if tb == 'patio' else 1          # abre hacia el interior
        elif ta in priv and tb not in priv:
            op['side'] = -1
        elif tb in priv and ta not in priv:
            op['side'] = 1


def floors(image, model, objs, rooms):
    """Acabado de piso por textura: madera (vetas en una dirección), cerámica (rejilla) o liso."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    lab = model['room_labels']
    occ = np.zeros(lab.shape, bool)
    for o in objs:
        occ[o['y0']:o['y1'], o['x0']:o['x1']] = True
    for r in rooms.values():
        if r['type'] == 'patio':
            r['floor'], r['floor_dir'] = 'exterior', 'h'
            continue
        m = (lab == r['label']) & ~occ
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if m.sum() < 30:
            r['floor'], r['floor_dir'] = 'tile' if r['type'] in ('bath', 'kitchen', 'laundry') else 'wood', 'h'
            continue
        ex, ey = np.median(gx[m]), np.median(gy[m])
        col = np.median(image[m], 0)  # BGR, robusto a muebles y alfombras
        warm = float(col[2]) - float(col[0])
        tex = (ex + ey) / 2
        if warm > 12:
            fl = 'wood'
        elif tex > 4 or r['type'] in ('bath', 'laundry'):
            fl = 'tile'
        else:
            fl = 'porcelain'
        r['floor'] = fl
        r['floor_dir'] = 'h' if ey > ex else 'v'
    return rooms


FLOOR_NAMES = {'wood': 'F1  Wood-look laminate / plank', 'tile': 'F2  Ceramic tile 60x60',
               'porcelain': 'F3  Porcelain / smooth finish'}


def electrical(model, objs, rooms):
    """Propuesta esquemática: luminarias por área, tomas cada ~3 m de perímetro (sin cruzar puertas),
    tomas de mesón cada ~1.2 m con GFCI, una toma GFCI junto al lavamanos, interruptor junto a cada puerta."""
    s = model['m_per_px']
    lab = model['room_labels']
    doors = [o for o in model['openings'] if o['kind'] == 'door']
    pts = []
    for r in rooms.values():
        m = (lab == r['label']).astype(np.uint8)
        x, y, w, h = r['bbox']
        # luminarias
        nl = max(1, int(round(r['area'] / 12)))
        if nl == 1:
            pts.append(dict(t='light', x=r['cx'], y=r['cy'], room=r['id']))
        else:
            cols = 2 if w >= h else 1
            rows_ = int(np.ceil(nl / cols)) if cols == 2 else nl
            if cols == 1:
                cols, rows_ = 1, nl
            for i in range(cols):
                for j in range(rows_):
                    pts.append(dict(t='light', x=x + (i + 0.5) * w / cols, y=y + (j + 0.5) * h / rows_, room=r['id']))
        if r['type'] in ('closet',):
            continue
        # tomas en el perímetro
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)[:, 0, :]
        step = (3.0 if r['type'] not in ('bath', 'hall', 'laundry', 'patio') else 99) / s
        acc, start = 0.0, 0.6 / s
        placed = 0
        for k in range(1, len(cnt)):
            acc += float(np.hypot(*(cnt[k] - cnt[k - 1])))
            if acc >= start + placed * step:
                px, py = cnt[k]
                near_door = any(d['x0'] - 0.5 / s <= px <= d['x1'] + 0.5 / s and d['y0'] - 0.5 / s <= py <= d['y1'] + 0.5 / s
                                for d in doors)
                if not near_door:
                    pts.append(dict(t='outlet', x=float(px), y=float(py), room=r['id']))
                placed += 1
        if r['type'] in ('hall', 'laundry', 'patio'):
            px, py = cnt[len(cnt) // 3]
            pts.append(dict(t='outlet', x=float(px), y=float(py), room=r['id']))
        # interruptores junto a las puertas del espacio
        for d in doors:
            cx, cy = (d['x0'] + d['x1']) / 2, (d['y0'] + d['y1']) / 2
            # el espacio al que da la puerta: muestra a cada lado
            off = 0.35 / s
            for sx, sy in ((cx, cy - off), (cx, cy + off)) if d['orient'] == 'h' else ((cx - off, cy), (cx + off, cy)):
                ix, iy = int(np.clip(sx, 0, lab.shape[1] - 1)), int(np.clip(sy, 0, lab.shape[0] - 1))
                if lab[iy, ix] == r['label']:
                    if d['orient'] == 'h':
                        pts.append(dict(t='switch', x=d['x1'] + 0.15 / s, y=sy, room=r['id']))
                    else:
                        pts.append(dict(t='switch', x=sx, y=d['y1'] + 0.15 / s, room=r['id']))
    # tomas de mesón y de lavamanos (GFCI)
    for o in objs:
        if o['kind'] == 'casework' and o.get('casework') in ('counter', 'island / bar'):
            L = max(o['w_m'], o['h_m'])
            n = max(1, int(L / 1.2))
            for i in range(n):
                f = (i + 0.5) / n
                if o['w_m'] >= o['h_m']:
                    pts.append(dict(t='gfci', x=o['x0'] + f * (o['x1'] - o['x0']), y=(o['y0'] + o['y1']) / 2, room=o['room']))
                else:
                    pts.append(dict(t='gfci', x=(o['x0'] + o['x1']) / 2, y=o['y0'] + f * (o['y1'] - o['y0']), room=o['room']))
        if o['kind'] == 'fixture' and o.get('fixture') == 'lavatory':
            pts.append(dict(t='gfci', x=o['x1'] + 0.25 / s, y=(o['y0'] + o['y1']) / 2, room=o['room']))
    return pts


def build(image, mask, model):
    objs = extract_objects(mask, model)
    rooms = classify(model, objs, image)
    doors(model, rooms)
    rooms = floors(image, model, objs, rooms)
    elec = electrical(model, objs, rooms)
    return dict(objects=objs, rooms=rooms, electrical=elec)
