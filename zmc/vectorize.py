"""Máscara (fondo/espacio/muro/puerta/ventana) -> modelo vectorial ortogonal en metros.

Solo reglas, sin IA:
 1. huella del apartamento = lo encerrado por la envolvente de muros (elimina fondos con patrón)
 2. muros = rectángulos horizontales/verticales (apertura morfológica con kernels largos)
 3. alineación de bordes a ejes comunes
 4. vanos (puertas/ventanas) asignados al muro que interrumpen
 5. espacios = componentes conexos del interior (las puertas cuentan como cierre)
 6. escala a partir del área privada declarada (m²)
"""
import json
import cv2
import numpy as np

BG, SP, WL, DR, WN, MS, AP, MU = range(8)


def _cc_boxes(binary, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n) if st[i, 4] >= min_area], lab, st


def main_envelope(mask):
    """Conserva solo la envolvente de muros más grande (descarta manchas en fondos y rótulos)."""
    t = wall_thickness(mask)
    env = np.isin(mask, (WL, DR, WN)).astype(np.uint8)
    k = max(3, int(round(t)) | 1)
    grown = cv2.dilate(env, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grown, 8)
    if n <= 1:
        return mask
    big = 1 + int(np.argmax(st[1:, 4]))
    keep = (lab == big)
    out = mask.copy()
    out[np.isin(mask, (WL, DR, WN)) & ~keep] = SP
    return out


def footprint(mask):
    env = np.isin(mask, (WL, DR, WN)).astype(np.uint8)
    t = wall_thickness(mask)
    k = max(3, int(round(t * 1.6)) | 1)
    closed = cv2.morphologyEx(env, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    h, w = closed.shape
    flood = closed.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 2)
    inside = (flood != 2).astype(np.uint8)  # envolvente + agujeros encerrados
    n, lab, st, _ = cv2.connectedComponentsWithStats(inside, 8)
    if n <= 1:
        return inside.astype(bool)
    big = 1 + int(np.argmax(st[1:, 4]))
    fp = lab == big
    return fp


def wall_thickness(mask):
    wl = (mask == WL).astype(np.uint8)
    if wl.sum() == 0:
        return 3.0
    dt = cv2.distanceTransform(wl, cv2.DIST_L2, 3)
    ridge = dt[(dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3)]
    return float(max(2.0, 2 * np.median(ridge)))


def bars(img, t, min_len):
    """Franjas horizontales -> rectángulos (x0, y0, x1, y1). Parte componentes escalonados
    agrupando columnas contiguas cuyo centro vertical coincide."""
    out = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < min_len:
            continue
        sub = (lab[y:y + h, x:x + w] == i)
        cnt = sub.sum(0)
        cy = (sub * np.arange(h)[:, None]).sum(0) / np.maximum(cnt, 1)
        groups, cur = [], [0]
        for j in range(1, w):
            if cnt[j] == 0 or cnt[cur[-1]] == 0 or abs(cy[j] - cy[cur[-1]]) > max(1.5, t * 0.5):
                groups.append(cur); cur = [j]
            else:
                cur.append(j)
        groups.append(cur)
        for g in groups:
            g = [j for j in g if cnt[j] > 0]
            if len(g) < min_len:
                continue
            j0, j1 = g[0], g[-1] + 1
            part = sub[:, j0:j1]
            rows = part.sum(1)
            keep = np.flatnonzero(rows >= 0.5 * rows.max())
            out.append((x + j0, y + int(keep[0]), x + j1, y + int(keep[-1]) + 1))
    return out


def snap(vals, tol):
    """Agrupa coordenadas cercanas y devuelve el mapa valor->valor alineado."""
    vs = sorted(set(vals))
    groups, cur = [], [vs[0]] if vs else []
    for v in vs[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    if cur:
        groups.append(cur)
    m = {}
    for g in groups:
        c = int(round(np.median(g)))
        for v in g:
            m[v] = c
    return m


def vectorize(mask, area_m2=None):
    mask = main_envelope(mask)
    fp = footprint(mask)
    mask[~fp] = BG
    t = wall_thickness(mask)
    solid = np.isin(mask, (WL, DR, WN)).astype(np.uint8)
    L = max(5, int(round(t * 2.5)))
    hor = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    walls = []
    tol = max(2, int(round(t * 0.75)))
    for (img, o) in ((hor, 'h'), (ver.T, 'v')):
        for (a, b, c, d) in bars(img, t, L):
            walls.append([o, a, b, c, d] if o == 'h' else [o, b, a, d, c])
    # tramos cortos que la apertura borró (esquinas, machones)
    cover = np.zeros_like(solid)
    for o, a, b, c, d in walls:
        cover[b:d, a:c] = 1
    rest = ((mask == WL) & (cover == 0)).astype(np.uint8)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((max(2, int(t * 0.6)),) * 2, np.uint8))
    for (x, y, w, h) in _cc_boxes(rest, int(t * t * 1.5))[0]:
        walls.append(['h' if w >= h else 'v', x, y, x + w, y + h])
    # alinear: ejes de muro (centro del grosor) y extremos, conservando el grosor
    tol = max(2, int(t * 0.6))
    cy = [(b + d) / 2 for o, a, b, c, d in walls if o == 'h']
    cx = [(a + c) / 2 for o, a, b, c, d in walls if o == 'v']
    ends_x = [a for o, a, b, c, d in walls if o == 'h'] + [c for o, a, b, c, d in walls if o == 'h'] + \
        [int(round(v)) for v in cx]
    ends_y = [b for o, a, b, c, d in walls if o == 'v'] + [d for o, a, b, c, d in walls if o == 'v'] + \
        [int(round(v)) for v in cy]
    mx = snap([int(round(v)) for v in ends_x], tol)
    my = snap([int(round(v)) for v in ends_y], tol)
    out = []
    for o, a, b, c, d in walls:
        if o == 'h':
            th = max(2, d - b); y = my.get(int(round((b + d) / 2)), int(round((b + d) / 2)))
            out.append([o, mx.get(a, a), y - th // 2, mx.get(c, c), y - th // 2 + th])
        else:
            th = max(2, c - a); x = mx.get(int(round((a + c) / 2)), int(round((a + c) / 2)))
            out.append([o, x - th // 2, my.get(b, b), x - th // 2 + th, my.get(d, d)])
    walls = [w for w in out if w[3] > w[1] and w[4] > w[2]]
    # vanos
    solid_w = np.isin(mask, (WL, WN, DR)).astype(np.float32) * 0 + (mask == WL)
    openings = []
    for cls, kind in ((DR, 'door'), (WN, 'window')):
        boxes, _, _ = _cc_boxes(mask == cls, max(4, int(t * t * 0.6)))
        for (x, y, w, h) in boxes:
            o = 'h' if w >= h else 'v'
            ln = max(w, h)
            if ln < 1.5 * t:
                continue
            # el vano debe estar entre muros: revisar ambos extremos sobre su eje
            e = int(round(t * 1.5))
            if o == 'h':
                ends = [solid_w[y:y + h, max(0, x - e):x], solid_w[y:y + h, x + w:x + w + e]]
            else:
                ends = [solid_w[max(0, y - e):y, x:x + w], solid_w[y + h:y + h + e, x:x + w]]
            hits = sum(int(a.size > 0 and a.mean() > 0.25) for a in ends)
            if hits < (2 if kind == 'door' else 1):
                continue
            openings.append(dict(kind=kind, orient=o, x0=x, y0=y, x1=x + w, y1=y + h))
    # espacios: interior sin muros ni vanos
    closure = np.zeros_like(solid)
    for o, a, b, c, d in walls:
        cv2.rectangle(closure, (a, b), (c - 1, d - 1), 1, -1)
    for op in openings:
        cv2.rectangle(closure, (op['x0'], op['y0']), (op['x1'] - 1, op['y1'] - 1), 1, -1)
    closure = cv2.dilate(closure, np.ones((3, 3), np.uint8))
    inner = fp & ~(closure > 0)
    inner = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(inner, 4)
    min_room = (t * 6) ** 2
    rooms = []
    for i in range(1, n):
        if st[i, 4] < min_room:
            continue
        rooms.append(dict(id=len(rooms) + 1, label=i, area_px=int(st[i, 4]), cx=float(cen[i, 0]), cy=float(cen[i, 1]),
                          bbox=[int(v) for v in st[i, :4]]))
    inner_px = sum(r['area_px'] for r in rooms) or int(fp.sum())   # sin espacios cerrados: usa la huella
    m_per_px = float(np.sqrt(area_m2 / inner_px)) if area_m2 and inner_px else None
    ys, xs = np.where(fp)
    return dict(thickness_px=t, m_per_px=m_per_px, area_m2=area_m2,
                bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                walls=walls, openings=openings, rooms=rooms, shape=list(mask.shape),
                footprint=fp, room_labels=lab)


def preview(model, size_hw):
    """Dibujo rápido de control del modelo vectorial."""
    h, w = size_hw
    img = np.full((h, w, 3), 255, np.uint8)
    for r in model['rooms']:
        pass
    for o, a, b, c, d in model['walls']:
        cv2.rectangle(img, (a, b), (c - 1, d - 1), (40, 40, 40), -1)
    for op in model['openings']:
        col = (0, 140, 255) if op['kind'] == 'door' else (255, 120, 40)
        cv2.rectangle(img, (op['x0'], op['y0']), (op['x1'] - 1, op['y1'] - 1), col, -1)
    for r in model['rooms']:
        s = f"{r['id']}"
        if model['m_per_px']:
            s += f" {r['area_px'] * model['m_per_px'] ** 2:.1f}"
        cv2.putText(img, s, (int(r['cx']) - 12, int(r['cy'])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1,
                    cv2.LINE_AA)
    return img
