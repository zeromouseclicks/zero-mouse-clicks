"""Render sintético en alta resolución para el banco de medidas (copia de zmc.synth.render sin rotación
ni textos de relleno, que devuelve los rásters de muros a resolución supermuestreada). Generado desde
zmc/synth.py; mantener en sintonía si cambia el generador."""
import numpy as np
import cv2
from zmc import synth as SY
from zmc.synth import make_layout, background, furniture, extras, tex_wood, tex_tile, tex_flat, SS


def render_hr(size=1600, style=None):
    style = style or SY.RNG.choice(['co', 'co', 'us', 'line'])
    L = make_layout()
    S = size * SS
    margin = SY.rnd(0.06, 0.22)
    px = (S * (1 - 2 * margin)) / max(L['W'], L['H'])  # píxeles por metro (supermuestreado)
    LAST = dict(layout=L, px_per_m=px / SS)   # para calcular el área real de la planta generada
    ox = (S - L['W'] * px) / 2 + SY.rnd(-0.05, 0.05) * S
    oy = (S - L['H'] * px) / 2 + SY.rnd(-0.05, 0.05) * S

    def P(x, y):
        return int(round(ox + x * px)), int(round(oy + y * px))

    img = background(S, S, style)
    mask = np.zeros((S, S), np.uint8)
    SY.OBJ = np.zeros((S, S), np.uint8)

    t_ext = SY.rnd(0.15, 0.25) * px          # banco: grosores de folleto real (ext. 15-25 cm, int. 8-15 cm)
    t_int = SY.rnd(0.08, 0.15) * px
    LAST.update(ox=ox / SS, oy=oy / SS, t_ext_px=max(3, int(t_ext)) / SS, t_int_px=2 * max(2, int(t_int / 2)) / SS, rot=0)
    _b = SY.rnd(105, 185); _g = _b + SY.rnd(8, 25)
    wood = [(_b, _g, min(250, _g + SY.rnd(8, 28)))]  # BGR: madera gris-café
    # pisos
    for r, t in zip(L['rooms'], L['types']):
        a, b = P(r[0], r[1])
        c, d = P(r[2], r[3])
        h, w = d - b, c - a
        if h <= 0 or w <= 0:
            continue
        if style == 'line':
            patch = np.full((h, w, 3), 255, np.float32)
        elif t in ('bath', 'laundry', 'kitchen') and SY.RNG.random() < 0.8:
            g = SY.rnd(200, 235)
            patch = tex_tile(h, w, (g, g, g + SY.rnd(-5, 5)))
        elif style == 'us' or SY.RNG.random() < 0.25:
            g = SY.rnd(185, 225)
            patch = tex_flat(h, w, (g, g, g)) if SY.RNG.random() < 0.5 else tex_wood(h, w, wood[0])
        else:
            patch = tex_wood(h, w, wood[0])
        img[b:d, a:c] = patch
        mask[b:d, a:c] = 1
    # muebles
    for r, t in zip(L['rooms'], L['types']):
        a, b = P(r[0], r[1])
        c, d = P(r[2], r[3])
        if SY.RNG.random() < 0.9:
            extras(img, (a + t_int, b + t_int, c - t_int, d - t_int), t, px, style)
            furniture(img, (a + t_int, b + t_int, c - t_int, d - t_int), t, px, style)

    # muros: polígonos rellenos
    wall_m = np.zeros((S, S), np.uint8)
    a, b = P(0, 0)
    # exterior: contorno de la unión de recintos
    union = np.zeros((S, S), np.uint8)
    for r in L['rooms']:
        p0, p1 = P(r[0], r[1]), P(r[2], r[3])
        cv2.rectangle(union, p0, p1, 255, -1)
    k = max(3, int(t_ext))
    ext = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) - \
        cv2.erode(union, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    wall_m |= ext
    ti = max(2, int(t_int / 2))
    for r in L['rooms']:
        p0, p1 = P(r[0], r[1]), P(r[2], r[3])
        cv2.rectangle(wall_m, p0, p1, 255, 2 * ti)
    wall_m &= cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # ventanas sobre el contorno exterior
    win_m = np.zeros((S, S), np.uint8)
    door_m = np.zeros((S, S), np.uint8)
    for r in L['rooms']:
        for side in range(4):
            on_border = [r[1] < 1e-6, r[2] > L['W'] - 1e-6, r[3] > L['H'] - 1e-6, r[0] < 1e-6][side]
            if not on_border or SY.RNG.random() < 0.3:
                continue
            if side in (0, 2):
                span = r[2] - r[0]
                ln = min(span - 0.6, SY.rnd(0.8, 2.6))
                if ln < 0.6:
                    continue
                s0 = r[0] + SY.rnd(0.3, span - ln - 0.3)
                y = r[1] if side == 0 else r[3]
                p0, p1 = P(s0, y), P(s0 + ln, y)
                cv2.rectangle(win_m, (p0[0], p0[1] - k // 2 - 1), (p1[0], p1[1] + k // 2 + 1), 255, -1)
            else:
                span = r[3] - r[1]
                ln = min(span - 0.6, SY.rnd(0.8, 2.6))
                if ln < 0.6:
                    continue
                s0 = r[1] + SY.rnd(0.3, span - ln - 0.3)
                x = r[2] if side == 1 else r[0]
                p0, p1 = P(x, s0), P(x, s0 + ln)
                cv2.rectangle(win_m, (p0[0] - k // 2 - 1, p0[1]), (p1[0] + k // 2 + 1, p1[1]), 255, -1)
    win_m &= wall_m

    # puertas interiores
    door_draw = []
    for (o, f, lo, hi) in L['doors']:
        dw = min(hi - lo - 0.2, SY.rnd(0.7, 0.95))
        s0 = SY.rnd(lo + 0.1, hi - dw - 0.1)
        if o == 'v':
            p0, p1 = P(f, s0), P(f, s0 + dw)
            cv2.rectangle(door_m, (p0[0] - ti - 2, p0[1]), (p1[0] + ti + 2, p1[1]), 255, -1)
        else:
            p0, p1 = P(s0, f), P(s0 + dw, f)
            cv2.rectangle(door_m, (p0[0], p0[1] - ti - 2), (p1[0], p1[1] + ti + 2), 255, -1)
        door_draw.append((o, p0, p1))
    # puerta de acceso en muro exterior
    ent = None
    for _ in range(20):
        r = L['rooms'][SY.RNG.integers(len(L['rooms']))]
        if r[1] < 1e-6 and r[2] - r[0] > 1.4:
            s0 = SY.rnd(r[0] + 0.2, r[2] - 1.1)
            p0, p1 = P(s0, 0), P(s0 + 0.9, 0)
            cv2.rectangle(door_m, (p0[0], p0[1] - k // 2 - 1), (p1[0], p1[1] + k // 2 + 1), 255, -1)
            door_draw.append(('h', p0, p1))
            ent = ((p0[0] + p1[0]) // 2, p0[1] - k)
            break
    door_m &= (wall_m | cv2.dilate(wall_m, np.ones((3, 3), np.uint8)))
    win_m[door_m > 0] = 0
    solid = (wall_m > 0) & (door_m == 0) & (win_m == 0)

    # dibujar muros según estilo
    if style == 'co':
        wc = SY.rnd(0, 40)
        img[solid] = wc
    elif style == 'us':
        sh = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 3 * SS)
        img *= (1 - 0.35 * np.roll(np.roll(sh, 3 * SS, 0), 3 * SS, 1))[..., None]
        wc = SY.rnd(225, 250)
        img[solid] = wc
        edges = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[edges] = SY.rnd(120, 170)
    else:
        img[solid] = SY.rnd(0, 60) if SY.RNG.random() < 0.5 else 255
        edges = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[edges] = 20
    # ventanas: varias convenciones gráficas
    wm = win_m > 0
    wv = SY.RNG.choice(['thin', 'glass', 'gray', 'gap'])
    wcol = (SY.rnd(150, 210), SY.rnd(150, 200), SY.rnd(140, 190))
    if style == 'line':
        wv = SY.RNG.choice(['thin', 'gap'])
    if wv == 'glass':  # vidrio azul claro con marco
        img[wm] = (SY.rnd(220, 250), SY.rnd(200, 230), SY.rnd(160, 200))
    elif wv == 'gray':  # banda gris clara
        img[wm] = SY.rnd(170, 215)
    else:
        img[wm] = (250, 250, 250) if style != 'line' else (255, 255, 255)
    we = cv2.morphologyEx(win_m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if wv != 'gap':
        img[we & wm] = wcol
    if wv == 'thin':  # línea central (doble línea)
        img[cv2.erode(win_m, np.ones((max(3, k // 2), max(3, k // 2)), np.uint8)) > 0] = \
            np.array(wcol) * 0.9 if SY.RNG.random() < 0.6 else (230, 230, 230)
    # puertas: hueco + hoja + arco
    for (o, p0, p1) in door_draw:
        if SY.RNG.random() < 0.65:
            r_ = abs((p1[0] - p0[0]) if o == 'h' else (p1[1] - p0[1]))
            sgn = 1 if SY.RNG.random() < 0.5 else -1
            col = (90, 90, 90)
            if o == 'h':
                cv2.line(img, p0, (p0[0], p0[1] + sgn * r_), col, SS, cv2.LINE_AA)
                cv2.ellipse(img, p0, (r_, r_), 0, 0 if sgn > 0 else -90, 90 if sgn > 0 else 0,
                            col, max(1, SS // 2), cv2.LINE_AA)
            else:
                cv2.line(img, p0, (p0[0] + sgn * r_, p0[1]), col, SS, cv2.LINE_AA)
                cv2.ellipse(img, p0, (r_, r_), 0, 0 if sgn > 0 else 90, 90 if sgn > 0 else 180,
                            col, max(1, SS // 2), cv2.LINE_AA)
    # flecha roja de acceso
    if ent and style != 'line' and SY.RNG.random() < 0.7:
        s = int(0.35 * px)
        pts = np.array([[ent[0] - s // 2, ent[1] - s], [ent[0] + s // 2, ent[1] - s], [ent[0], ent[1]]])
        cv2.fillPoly(img, [pts], (30, 30, 220), cv2.LINE_AA)
    # textos, rótulos y logos fuera del apartamento
    img = np.clip(img, 0, 255).astype(np.uint8)
    outside = cv2.dilate((mask > 0).astype(np.uint8) | (wall_m > 0).astype(np.uint8),
                         np.ones((9, 9), np.uint8)) == 0
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN,
             cv2.FONT_HERSHEY_COMPLEX]

    def free(x0, y0, x1, y1):
        if x0 < 0 or y0 < 0 or x1 >= S or y1 >= S:
            return False
        return outside[y0:y1, x0:x1].all()

    # máscara final
    obj_in = (mask == 1) & (SY.OBJ > 0)
    mask[obj_in] = SY.OBJ[obj_in]
    mask[wall_m > 0] = 2
    mask[(door_m > 0)] = 3
    mask[(win_m > 0)] = 4
    img = np.clip(img, 0, 255).astype(np.uint8)
    hr = dict(img=img, wall=wall_m > 0, door=door_m > 0, win=win_m > 0, S=S, px=px, P=P, layout=L, LAST=LAST)
    return hr
