"""Reglas de limpieza entre el modelo y el vectorizador (v5, sin IA):
- recorte automático a la planta (plan_bbox)
- filtro de color de muro: lo que el modelo llama muro/puerta pero no tiene el color de los muros se descarta
"""
import cv2
import numpy as np
from .vectorize import BG, SP, WL, DR, WN, MS, AP, MU, main_envelope, footprint, wall_thickness


def plan_bbox(mask, margin=0.05):
    """Caja (x0, y0, x1, y1) de la planta dentro de la máscara, con margen.
    v0.5.1: también entran los pedazos de muro que quedaron sueltos de la envolvente principal (una esquina unida al
    resto solo por una ventana o por la puerta de entrada que el modelo no vio): antes se recortaban y la planta perdía
    muros (Rivera del Valle: dos muros de la sala)."""
    m = main_envelope(mask.copy())
    fp = footprint(m)
    ys, xs = np.where(fp)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    x0, y0, x1, y1 = _loose_walls(mask, (x0, y0, x1, y1))
    mg = int(round(margin * max(x1 - x0, y1 - y0))) + 4
    h, w = mask.shape
    return max(0, x0 - mg), max(0, y0 - mg), min(w, x1 + mg), min(h, y1 + mg)


def _loose_walls(mask, box, reach=0.18):
    """Agranda la caja con componentes de muro cercanos y con forma de muro (largos y delgados, o en L)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    t = wall_thickness(mask)
    env = np.isin(mask, (WL, DR, WN)).astype(np.uint8)
    k = max(3, int(round(t)) | 1)
    grown = cv2.dilate(env, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grown, 8)
    if n <= 2:
        return x0, y0, x1, y1
    wl_count = np.bincount(lab[mask == WL].ravel(), minlength=n)
    done = set()
    for _ in range(4):                       # en cadena: un pedazo puede quedar cerca solo de otro pedazo
        grew = False
        side = max(x1 - x0, y1 - y0)
        for i in range(1, n):
            if i in done:
                continue
            bx, by, bw, bh, area = (int(v) for v in st[i])
            if bx >= x0 and by >= y0 and bx + bw <= x1 and by + bh <= y1:
                done.add(i); continue
            dx = max(0, x0 - (bx + bw), bx - x1)
            dy = max(0, y0 - (by + bh), by - y1)
            if max(dx, dy) > reach * side:
                continue
            L, S = max(bw, bh), min(bw, bh)
            if L < max(0.12 * side, 6 * t):  # muy corto para ser un muro de la planta
                continue
            if wl_count[i] < 0.15 * area:        # casi nada de muro adentro (texto, mueble, mancha)
                continue
            if not (S <= 4 * t or area / float(bw * bh) < 0.4):   # ni delgado ni en L: un bloque (texto, logo)
                continue
            x0, y0, x1, y1 = min(x0, bx), min(y0, by), max(x1, bx + bw), max(y1, by + bh)
            done.add(i); grew = True
        if not grew:
            break
    return x0, y0, x1, y1


def cut_sides(mask, crop, shape):
    """Lados del recorte donde hay muro pegado al borde (y el borde no es el de la imagen): ahí se cortó la planta.
    El recorte deja un margen del 5 %, así que un muro a pocos píxeles del borde solo puede ser un muro cortado.
    El modelo casi nunca marca muro en los 2-3 px del borde, por eso se mira una franja de ~1,5 espesores."""
    h, w = mask.shape
    t = wall_thickness(mask)
    wl = mask == WL
    b = max(4, int(round(1.5 * t)))
    need = 0.8 * t * b                       # al menos un muro que atraviesa la franja
    cx0, cy0, cx1, cy1 = crop
    H, W = shape[:2]
    out = []
    if cx0 > 0 and wl[:, :b].sum() >= need:
        out.append('l')
    if cy0 > 0 and wl[:b, :].sum() >= need:
        out.append('t')
    if cx1 < W and wl[:, w - b:].sum() >= need:
        out.append('r')
    if cy1 < H and wl[h - b:, :].sum() >= need:
        out.append('b')
    return out


def wall_color_model(im, mask):
    """Color típico de los muros (Lab) si es consistente; si no, None."""
    wl = (mask == WL).astype(np.uint8)
    core = cv2.erode(wl, np.ones((3, 3), np.uint8)) > 0
    if core.sum() < 150:
        core = wl > 0
    if core.sum() < 150:
        return None
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[core]
    med = np.median(px, 0)
    d = np.linalg.norm(px - med, axis=1)
    tol = 38.0
    frac = float((d < tol).mean())
    return dict(med=med, tol=tol, frac=frac, lab=lab) if frac >= 0.55 else None


def wall_color_filter(im, mask, info=None):
    """Quita del muro (y de puertas/ventanas no hace falta) lo que no tiene color de muro."""
    cm = wall_color_model(im, mask)
    if info is not None:
        info['wall_color'] = None if cm is None else dict(med=[float(v) for v in cm['med']], frac=cm['frac'])
    if cm is None:
        return mask, None
    dist = np.linalg.norm(cm['lab'] - cm['med'], axis=2)
    like = (dist < cm['tol']).astype(np.uint8)
    like_d = cv2.dilate(like, np.ones((3, 3), np.uint8)) > 0     # bordes suavizados del muro
    out = mask.copy()
    bad = (mask == WL) & ~like_d
    out[bad] = SP
    return out, like > 0
