"""Reglas de limpieza entre el modelo y el vectorizador (v5, sin IA):
- recorte automático a la planta (plan_bbox)
- filtro de color de muro: lo que el modelo llama muro/puerta pero no tiene el color de los muros se descarta
"""
import cv2
import numpy as np
from .vectorize import BG, SP, WL, DR, WN, MS, AP, MU, main_envelope, footprint, wall_thickness


def plan_bbox(mask, margin=0.05):
    """Caja (x0, y0, x1, y1) de la planta dentro de la máscara, con margen."""
    m = main_envelope(mask.copy())
    fp = footprint(m)
    ys, xs = np.where(fp)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mg = int(round(margin * max(x1 - x0, y1 - y0))) + 4
    h, w = mask.shape
    return max(0, x0 - mg), max(0, y0 - mg), min(w, x1 + mg), min(h, y1 + mg)


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
