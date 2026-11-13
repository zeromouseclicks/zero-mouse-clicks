"""Escala del dibujo a partir de TODAS las evidencias (v6, solo reglas, sin tokens).

Antes la escala salía solo del área declarada (área = suma de espacios netos). Si el área venía con otra
definición (construida en vez de privada, sq ft brutos) o el OCR leía la equivocada, todo el plano quedaba
5-12 % más grande o más pequeño, aunque el folleto trajera las medidas de cada espacio impresas.

Ahora:
1. Medidas impresas («2.69 x 2.20», «12'4" x 11'0"») -> escala por votación (tipo RANSAC): cada medida se
   compara con varias longitudes medidas en la máscara (caja del espacio, rayos de cara a cara de muro,
   con y sin clósets/mesones) en los dos ejes; gana la escala que explica más medidas dentro de ±3,5 %.
   Si el folleto está estirado (eje X distinto del Y) y las medidas lo demuestran, se corrige.
2. Sin medidas: área declarada con su definición: privada ≈ espacios netos + media partición interior;
   construida / total / sq ft ≈ huella bruta (con muros).
3. Siempre se informa cuánto cuadran las evidencias entre sí (precisión estimada).
"""
import numpy as np
import cv2
from .vectorize import WL, DR, WN, MS

TOL = 0.035          # tolerancia relativa para que una medida impresa «cuadre»


def obstacles(model, mask):
    """Rásters de obstáculo para medir luces libres: (muros+vanos, muros+vanos+clósets/mesones)."""
    H, W = model['shape']
    A = np.zeros((H, W), np.uint8)
    for o, a, b, c, d in model['walls']:
        A[max(0, b):d, max(0, a):c] = 1
    for op in model['openings']:
        A[max(0, op['y0']):op['y1'], max(0, op['x0']):op['x1']] = 1
    B = A.copy()
    if mask is not None:
        cw = (mask == MS).astype(np.uint8)
        cw = cv2.morphologyEx(cw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        B[cw > 0] = 1
    return A, B


def _ray(obs, x, y, dx, dy, lim):
    """Distancia en px hasta el primer obstáculo desde (x, y) en la dirección (dx, dy); None si sale."""
    H, W = obs.shape
    for i in range(1, lim):
        xx, yy = x + dx * i, y + dy * i
        if xx < 0 or yy < 0 or xx >= W or yy >= H:
            return None
        if obs[yy, xx]:
            return i
    return None


def _snap(like, x, y, dx, dy, i, r):
    """Afina un impacto de rayo (a i px) con el color de muro de la imagen: primer píxel «color muro»
    dentro de ±r px alrededor del impacto (la cara real del muro). Sin color de muro, deja i."""
    if like is None or i is None:
        return i
    H, W = like.shape
    best = None
    for j in range(max(1, i - r), i + r + 1):
        xx, yy = x + dx * j, y + dy * j
        x2, y2 = xx + dx, yy + dy
        xp, yp = xx - dx, yy - dy
        if not (0 <= x2 < W and 0 <= y2 < H and 0 <= xp < W and 0 <= yp < H and 0 <= xx < W and 0 <= yy < H):
            continue
        if like[yy, xx] and like[y2, x2] and not like[yp, xp]:          # borde piso -> muro
            if best is None or abs(j - i) < abs(best - i):
                best = j
    return i if best is None else best


def spans(model, obsA, obsB, room, x, y, like=None):
    """Longitudes candidatas (px, de cara a cara) en cada eje para el espacio `room` alrededor de (x, y)."""
    bx, by, bw, bh = room['bbox']
    out = {'h': [bw + 2.0], 'v': [bh + 2.0]}       # +2: el espacio se encoge 1 px por lado al segmentar
    lim = int(max(model['shape']))
    r = max(2, int(round(0.6 * model['thickness_px'])))
    for obs in (obsA, obsB):
        for f in (-0.25, 0.0, 0.25):
            yy = int(round(y + f * bh)); xx = int(round(x + f * bw)); x0, y0 = int(round(x)), int(round(y))
            if by <= yy < by + bh:
                l, rr = _ray(obs, x0, yy, -1, 0, lim), _ray(obs, x0, yy, 1, 0, lim)
                if l and rr:
                    out['h'].append(float(l + rr - 1))
                    l2, r2 = _snap(like, x0, yy, -1, 0, l, r), _snap(like, x0, yy, 1, 0, rr, r)
                    if (l2, r2) != (l, rr):
                        out['h'].append(float(l2 + r2 - 1))
            if bx <= xx < bx + bw:
                u, d = _ray(obs, xx, y0, 0, -1, lim), _ray(obs, xx, y0, 0, 1, lim)
                if u and d:
                    out['v'].append(float(u + d - 1))
                    u2, d2 = _snap(like, xx, y0, 0, -1, u, r), _snap(like, xx, y0, 0, 1, d, r)
                    if (u2, d2) != (u, d):
                        out['v'].append(float(u2 + d2 - 1))
    # sin duplicados casi iguales
    for k in out:
        vals = sorted(out[k]); keep = []
        for v in vals:
            if not keep or v > keep[-1] * 1.01:
                keep.append(v)
        out[k] = keep
    return out


def _match(v, cands, s):
    """Mejor error relativo log de la medida v (m) contra longitudes cands (px) a escala s (m/px)."""
    if not cands:
        return None, None
    e = [abs(np.log(v / (L * s))) for L in cands]
    i = int(np.argmin(e))
    return e[i], cands[i]


def fit_dims(model, mask, dims, k, room_at, like=None):
    """Escala (sx, sy) en m/px de la máscara a partir de las medidas impresas. dims: [{a, b, x, y}] en coordenadas
    de la imagen leída; k: px de la máscara por px leído. Devuelve None si no hay evidencia suficiente."""
    if not dims:
        return None
    obsA, obsB = obstacles(model, mask)
    by_label = {r['label']: r for r in model['rooms']}
    obs = []                                          # una entrada por par impreso
    for d in dims:
        x, y = d['x'] * k, d['y'] * k
        r = by_label.get(room_at(model['room_labels'], x, y, int(2.5 * model['thickness_px'])))
        if r is None:
            continue
        sp = spans(model, obsA, obsB, r, x, y, like)
        w = 0.5 if r.get('exterior') else 1.0
        obs.append(dict(a=d['a'], b=d['b'], sp=sp, room=r['id'], w=w))
    if not obs:
        return None

    def score(sx, sy):
        # cada espacio aporta a lo sumo un par impreso (si varios rótulos caen en el mismo espacio detectado,
        # la segmentación unió espacios y esas luces no son confiables)
        per = {}
        for o in obs:
            best = (0, None)
            for va, vb, conv in ((o['a'], o['b'], 'a=v'), (o['b'], o['a'], 'a=h')):   # va en vertical
                m = []
                ev, Lv = _match(va, o['sp']['v'], sy)
                eh, Lh = _match(vb, o['sp']['h'], sx)
                if ev is not None and ev <= TOL:
                    m.append(('v', va, Lv, ev))
                if eh is not None and eh <= TOL:
                    m.append(('h', vb, Lh, eh))
                if len(m) > best[0] or (len(m) == best[0] and m and best[1] and sum(q[3] for q in m) < sum(q[3] for q in best[1])):
                    best = (len(m), m)
            if best[1]:
                cur = per.get(o['room'])
                err = sum(q[3] for q in best[1])
                if cur is None or best[0] > cur[0] or (best[0] == cur[0] and err < cur[1]):
                    per[o['room']] = (best[0], err, [(o['room'], o['w']) + q for q in best[1]], o['w'])
        tot = sum(v[0] * v[3] for v in per.values())
        used = [u for v in per.values() for u in v[2]]
        return tot, used

    # hipótesis: isotrópicas desde cada valor y candidato; anisotrópicas desde cada par completo
    hyps = set()
    for o in obs:
        for va, vb in ((o['a'], o['b']), (o['b'], o['a'])):
            for Lv in o['sp']['v']:
                hyps.add((va / Lv, va / Lv))
                for Lh in o['sp']['h']:
                    hyps.add((vb / Lh, va / Lv))
            for Lh in o['sp']['h']:
                hyps.add((vb / Lh, vb / Lh))
    best = None
    for sx, sy in hyps:
        if not (0.2 < sx / sy < 5):
            continue
        iso = abs(np.log(sx / sy)) < 1e-9
        sc, used = score(sx, sy)
        if not iso:
            sc -= 1.5                                    # una distorsión hay que demostrarla
        if best is None or sc > best[0] + 1e-9 or (abs(sc - best[0]) < 1e-9 and np.mean([u[5] for u in used] or [1]) < best[3]):
            best = (sc, (sx, sy), used, np.mean([u[5] for u in used] or [1]))
    if best is None:
        return None
    sc, (sx, sy), used, _ = best
    rooms = {u[0] for u in used if u[1] >= 1.0}
    n = len(used)
    if not (n >= 3 and len(rooms) >= 2):
        return None
    # refinar: mediana de cada eje con las medidas que cuadran (pesos: interiores primero)
    def refine(sx, sy):
        _, used = score(sx, sy)
        hx = [u[3] / u[4] for u in used if u[2] == 'h']
        vy = [u[3] / u[4] for u in used if u[2] == 'v']
        return used, hx, vy
    used, hx, vy = refine(sx, sy)
    aniso = abs(np.log(sx / sy)) > 1e-9
    if aniso and len(hx) >= 2 and len(vy) >= 2:
        sx, sy = float(np.median(hx)), float(np.median(vy))
        if abs(np.log(sx / sy)) < 0.02:                  # distorsión despreciable: isotrópica
            aniso = False
    if not aniso or len(hx) < 2 or len(vy) < 2:
        aniso = False
        s = float(np.median(hx + vy)); sx = sy = s
    used, hx, vy = refine(sx, sy)
    res = [np.log(u[3] / (u[4] * (sx if u[2] == 'h' else sy))) for u in used]
    return dict(sx=sx, sy=sy, aniso=aniso, n=len(used), n_rooms=len({u[0] for u in used}), n_dims=len(obs),
                n_values=2 * len(obs), spread=float(np.percentile(np.abs(res), 90)) if res else None,
                used=[dict(room=u[0], axis=u[2], value=u[3], px=u[4]) for u in used])


# ----------------------------------------------------------------- áreas
def regrow(model, steps=3):
    """Áreas netas de cada espacio llevadas hasta la cara del muro (la segmentación las encoge ~1 px por lado).
    Devuelve {id: px}."""
    lab = model['room_labels']
    A, _ = obstacles(model, None)
    free = model['footprint'] & (A == 0)
    cur = lab.copy().astype(np.int32)
    keep = {r['label'] for r in model['rooms']}
    cur[~np.isin(cur, list(keep))] = 0
    for _ in range(steps):
        d = cv2.dilate(cur.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        g = (cur == 0) & free & (d > 0)
        if not g.any():
            break
        cur[g] = d[g]
    cnt = np.bincount(cur.ravel(), minlength=int(lab.max()) + 1)
    return {r['id']: int(cnt[r['label']]) for r in model['rooms']}


def area_terms(model, image=None):
    """Áreas en px² según la definición: neta, privada (neta + media partición interior) y bruta (huella
    con muros, sin exteriores verdes)."""
    from .vectorize5 import green_mask
    net = model.get('net_px') or regrow(model)
    inter = [r for r in model['rooms'] if not r.get('exterior')]
    net_px = float(sum(net[r['id']] for r in inter)) or float(model['footprint'].sum())
    fp = model['footprint'].copy()
    if image is not None:
        fp &= ~green_mask(image)
    lab = model['room_labels']
    outside = ~model['footprint']
    for r in model['rooms']:
        if r.get('exterior'):
            outside |= (lab == r['label'])
            fp &= ~(lab == r['label'])
    A, _ = obstacles(model, None)
    t = model['thickness_px']
    dist = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 3)
    interior_walls = float(((A > 0) & model['footprint'] & (dist > 1.6 * t)).sum())
    return dict(net=net_px, private=net_px + 0.5 * interior_walls, gross=float(fp.sum()))


def scale_from_area(area_m2, kind, terms):
    """m/px para un área declarada de cierta definición ('private' | 'gross' | 'net')."""
    px = terms.get(kind) or terms['private']
    return float(np.sqrt(area_m2 / px)) if area_m2 and px else None


def classify_area(area_m2, s, terms):
    """¿A qué definición corresponde un área declarada, dada una escala confiable? -> (tipo, desvío relativo)."""
    best = None
    for kind in ('private', 'gross'):
        a = terms[kind] * s * s
        e = area_m2 / a - 1
        if best is None or abs(e) < abs(best[1]):
            best = (kind, e)
    return best
