"""Zero Mouse Clicks — generador sintético de plantas estilo folleto.

Crea apartamentos aleatorios con reglas y los dibuja como plantas de mercadeo,
con la máscara exacta por píxel:
  0 fondo · 1 espacio · 2 muro · 3 puerta · 4 ventana · 5 mesón/closet/barra · 6 aparato fijo · 7 mueble suelto
Estilos: 'co' (muros negros, render a color, tipo constructora colombiana),
         'us' (render 3D cenital, muros claros con sombra, tipo Zillow),
         'line' (dibujo lineal blanco y negro).
"""
import cv2
import numpy as np

RNG = np.random.default_rng()
LAST = None  # distribución y escala de la última planta generada
SS = 2  # supermuestreo para antialias


def rnd(a, b):
    return RNG.uniform(min(a, b), max(a, b))


# ---------------------------------------------------------------- distribución
def split_rooms(x0, y0, x1, y1, min_side, depth=0, out=None):
    """BSP: parte el rectángulo en recintos."""
    if out is None:
        out = []
    w, h = x1 - x0, y1 - y0
    can_v, can_h = w >= 2 * min_side, h >= 2 * min_side
    stop = depth >= 2 and RNG.random() < 0.25 + 0.15 * depth
    if (not can_v and not can_h) or stop or w * h < 9:
        out.append([x0, y0, x1, y1])
        return out
    vert = can_v and (not can_h or (w > h if RNG.random() < 0.8 else RNG.random() < 0.5))
    if vert:
        c = rnd(x0 + min_side, x1 - min_side)
        split_rooms(x0, y0, c, y1, min_side, depth + 1, out)
        split_rooms(c, y0, x1, y1, min_side, depth + 1, out)
    else:
        c = rnd(y0 + min_side, y1 - min_side)
        split_rooms(x0, y0, x1, c, min_side, depth + 1, out)
        split_rooms(x0, c, x1, y1, min_side, depth + 1, out)
    return out


def shared_edge(a, b, eps=1e-6):
    """Segmento común entre dos rectángulos adyacentes: (orient, fijo, ini, fin)."""
    for (pa, pb) in ((a[2], b[0]), (b[2], a[0])):
        if abs(pa - pb) < eps:
            lo, hi = max(a[1], b[1]), min(a[3], b[3])
            if hi - lo > 0.9:
                return ('v', pa, lo, hi)
    for (pa, pb) in ((a[3], b[1]), (b[3], a[1])):
        if abs(pa - pb) < eps:
            lo, hi = max(a[0], b[0]), min(a[2], b[2])
            if hi - lo > 0.9:
                return ('h', pa, lo, hi)
    return None


def make_layout():
    W, H = rnd(6.5, 15.0), rnd(5.5, 10.5)
    rooms = split_rooms(0, 0, W, H, min_side=rnd(1.6, 2.4))
    # recorte en L: quitar un recinto de esquina a veces
    if len(rooms) > 4 and RNG.random() < 0.4:
        corners = [i for i, r in enumerate(rooms)
                   if (r[0] < 1e-6 or r[2] > W - 1e-6) and (r[1] < 1e-6 or r[3] > H - 1e-6)]
        if corners:
            rooms.pop(int(RNG.choice(corners)))
    n = len(rooms)
    adj = {}
    for i in range(n):
        for j in range(i + 1, n):
            e = shared_edge(rooms[i], rooms[j])
            if e:
                adj[(i, j)] = e
    # árbol de conexiones (puertas) + algunas aberturas extra
    big = int(np.argmax([(r[2] - r[0]) * (r[3] - r[1]) for r in rooms]))
    seen, doors, frontier = {big}, [], [big]
    while frontier:
        k = frontier.pop(RNG.integers(len(frontier)))
        nbrs = [(p, e) for p, e in adj.items() if k in p]
        RNG.shuffle(nbrs)
        for p, e in nbrs:
            o = p[0] if p[1] == k else p[1]
            if o not in seen:
                seen.add(o)
                frontier.append(o)
                doors.append(e)
    for p, e in adj.items():
        if RNG.random() < 0.12:
            doors.append(e)
    # tipos de recinto
    areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rooms]
    types = []
    for i, a in enumerate(areas):
        if i == big:
            types.append('living')
        elif a < 5.5:
            types.append(RNG.choice(['bath', 'bath', 'laundry', 'closet']))
        elif a < 9:
            types.append(RNG.choice(['bath', 'kitchen', 'bed', 'study']))
        else:
            types.append(RNG.choice(['bed', 'bed', 'kitchen', 'living2']))
    return dict(W=W, H=H, rooms=rooms, doors=doors, types=types)


# ---------------------------------------------------------------- texturas
def tex_wood(h, w, base):
    img = np.zeros((h, w, 3), np.float32) + np.array(base, np.float32)
    plank = int(rnd(8, 20)) * SS
    horiz = RNG.random() < 0.5
    n = (h if horiz else w) // plank + 1
    for k in range(n):
        shade = rnd(-18, 18)
        sl = slice(k * plank, (k + 1) * plank)
        if horiz:
            img[sl] += shade
        else:
            img[:, sl] += shade
    grain = cv2.GaussianBlur(RNG.normal(0, 10, (h, w)).astype(np.float32),
                             (1, 31) if horiz else (31, 1), 0)
    img += grain[..., None]
    return img


def tex_tile(h, w, base):
    img = np.zeros((h, w, 3), np.float32) + np.array(base, np.float32)
    s = int(rnd(10, 30)) * SS
    line = np.array(base, np.float32) - rnd(15, 35)
    img[::s] = line
    img[:, ::s] = line
    img += RNG.normal(0, 3, (h, w, 1)).astype(np.float32)
    return img


def tex_flat(h, w, base):
    img = np.zeros((h, w, 3), np.float32) + np.array(base, np.float32)
    img += cv2.GaussianBlur(RNG.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 6)[..., None]
    return img


def background(h, w, style):
    r = RNG.random()
    if style == 'line' or r < 0.45:
        bg = np.full((h, w, 3), rnd(238, 255), np.float32)
    elif r < 0.75:  # patrones: cuadros redondeados (típicos de folletos de constructoras), puntos, rayas, rejilla
        g = rnd(215, 245)
        bg = np.full((h, w, 3), g, np.float32)
        s = int(rnd(0.025, 0.075) * w)
        c = g + (rnd(-28, -6) if RNG.random() < 0.75 else rnd(5, 10))
        t = max(1, int(rnd(1, 3.5) * SS))
        kind = RNG.choice(['rsq', 'rsq', 'rsq', 'dots', 'diag', 'grid'])
        gap = int(s * rnd(0.12, 0.3))
        rr = int(s * rnd(0.1, 0.3))
        for yy in range(0, h, s):
            for xx in range(0, w, s):
                if kind == 'rsq':
                    a, b, c2, d = xx + gap, yy + gap, xx + s - gap, yy + s - gap
                    for (p, q) in (((a + rr, b), (c2 - rr, b)), ((a + rr, d), (c2 - rr, d)),
                                   ((a, b + rr), (a, d - rr)), ((c2, b + rr), (c2, d - rr))):
                        cv2.line(bg, p, q, (c, c, c), t, cv2.LINE_AA)
                    for (cx, cy, ang) in ((a + rr, b + rr, 180), (c2 - rr, b + rr, 270),
                                          (c2 - rr, d - rr, 0), (a + rr, d - rr, 90)):
                        cv2.ellipse(bg, (cx, cy), (rr, rr), ang, 0, 90, (c, c, c), t, cv2.LINE_AA)
                elif kind == 'dots':
                    cv2.circle(bg, (xx + s // 2, yy + s // 2), max(1, s // 8), (c, c, c), -1, cv2.LINE_AA)
        if kind == 'diag':
            for k in range(-h, w, s):
                cv2.line(bg, (k, 0), (k + h, h), (c, c, c), t, cv2.LINE_AA)
        elif kind == 'grid':
            bg[::s] = c
            bg[:, ::s] = c
    else:
        g = rnd(200, 245)
        bg = np.full((h, w, 3), g, np.float32)
        yy = np.linspace(-1, 1, h)[:, None]
        xx = np.linspace(-1, 1, w)[None]
        bg -= (rnd(10, 40) * (xx ** 2 + yy ** 2))[..., None]
    return bg


# ---------------------------------------------------------------- muebles
# ---------------------------------------------------------------- objetos (v4)
MESON, APARATO, MUEBLE = 5, 6, 7  # mesón/closet/barra · aparato fijo (sanitario/cocina) · mueble suelto
OBJ = None  # máscara de objetos de la planta que se está dibujando


def orect(a, b, c, d, cls):
    if OBJ is not None:
        cv2.rectangle(OBJ, (int(min(a, c)), int(min(b, d))), (int(max(a, c)), int(max(b, d))), cls, -1)


def ocircle(c, r, cls):
    if OBJ is not None:
        cv2.circle(OBJ, (int(c[0]), int(c[1])), int(r), cls, -1)


def oellipse(c, ax, cls):
    if OBJ is not None:
        cv2.ellipse(OBJ, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), 0, 0, 360, cls, -1)


def furniture(img, rect, kind, px, style):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if w < 30 or h < 30:
        return
    pal = [(225, 225, 230), (205, 215, 225), (170, 190, 200), (200, 185, 160),
           (150, 175, 160), (230, 215, 200), (120, 130, 140)]
    col = pal[RNG.integers(len(pal))]
    if style == 'line':
        col = (255, 255, 255)
    edge = (60, 60, 60) if style == 'line' else (110, 110, 110)
    m = 0.15 * px

    def box(a, b, c, d, color, cls=MUEBLE):
        a, b, c, d = int(a), int(b), int(c), int(d)
        orect(a, b, c, d, cls)
        if style != 'line':  # sombra
            cv2.rectangle(img, (a + 3 * SS, b + 3 * SS), (c + 3 * SS, d + 3 * SS),
                          tuple(float(v) * 0.75 for v in img[min(b, img.shape[0] - 1),
                                                              min(a, img.shape[1] - 1)]), -1)
        cv2.rectangle(img, (a, b), (c, d), color, -1)
        cv2.rectangle(img, (a, b), (c, d), edge, SS, lineType=cv2.LINE_AA)

    if kind in ('bed',):
        bw, bh = 1.5 * px, 2.0 * px
        if bw < w - 2 * m and bh < h - 2 * m:
            a = x0 + m + rnd(0, w - bw - 2 * m)
            box(a, y0 + m, a + bw, y0 + m + bh, col)
            cv2.rectangle(img, (int(a + 0.08 * px), int(y0 + m + 0.08 * px)),
                          (int(a + bw / 2 - 0.05 * px), int(y0 + m + 0.4 * px)), (245, 245, 245), -1)
            cv2.rectangle(img, (int(a + bw / 2 + 0.05 * px), int(y0 + m + 0.08 * px)),
                          (int(a + bw - 0.08 * px), int(y0 + m + 0.4 * px)), (245, 245, 245), -1)
    elif kind in ('living', 'living2'):
        sw, sh = min(2.2 * px, w * 0.6), 0.85 * px
        a = x0 + m + rnd(0, max(1, w - sw - 2 * m))
        b = y1 - m - sh
        box(a, b, a + sw, b + sh, col)
        if RNG.random() < 0.7:  # tapete y mesa
            ra = (a + 0.2 * px, b - 1.6 * px, a + sw - 0.2 * px, b - 0.1 * px)
            if ra[1] > y0 + m:
                cv2.rectangle(img, tuple(map(int, ra[:2])), tuple(map(int, ra[2:])),
                              (rnd(200, 235), rnd(195, 225), rnd(170, 210)), -1)
        if RNG.random() < 0.8 and w > 3.5 * px:  # comedor
            cx, cy = x1 - m - 0.9 * px, y0 + m + 0.9 * px
            box(cx - 0.6 * px, cy - 0.45 * px, cx + 0.6 * px, cy + 0.45 * px, (170, 130, 95))
            for dx in (-0.35, 0.35):
                for dy in (-0.65, 0.65):
                    cv2.circle(img, (int(cx + dx * px), int(cy + dy * px)), int(0.18 * px),
                               (70, 70, 70) if style != 'line' else (255, 255, 255), -1)
                    ocircle((cx + dx * px, cy + dy * px), 0.18 * px, MUEBLE)
                    cv2.circle(img, (int(cx + dx * px), int(cy + dy * px)), int(0.18 * px), edge, SS)
    elif kind == 'bath':
        cx = x0 + m + 0.3 * px
        cy = y0 + m + 0.35 * px
        cv2.ellipse(img, (int(cx), int(cy)), (int(0.2 * px), int(0.28 * px)), 0, 0, 360,
                    (250, 250, 250), -1)
        oellipse((cx, cy), (0.2 * px, 0.28 * px), APARATO)
        cv2.ellipse(img, (int(cx), int(cy)), (int(0.2 * px), int(0.28 * px)), 0, 0, 360, edge, SS)
        box(x1 - m - 0.6 * px, y0 + m, x1 - m, y0 + m + 0.45 * px, (245, 245, 245), APARATO)
    elif kind == 'kitchen':
        d = 0.6 * px
        horiz = w > h
        if horiz:
            box(x0 + m, y0 + m, x1 - m, y0 + m + d, (rnd(80, 220),) * 3, MESON)
            cv2.rectangle(img, (int(x0 + m + 0.5 * px), int(y0 + m + 0.1 * px)),
                          (int(x0 + m + 1.0 * px), int(y0 + m + 0.5 * px)), (190, 190, 195), -1)
            orect(x0 + m + 0.5 * px, y0 + m + 0.1 * px, x0 + m + 1.0 * px, y0 + m + 0.5 * px, APARATO)
        else:
            box(x0 + m, y0 + m, x0 + m + d, y1 - m, (rnd(80, 220),) * 3, MESON)
    elif kind == 'study':
        box(x0 + m, y0 + m, x0 + m + 1.2 * px, y0 + m + 0.6 * px, (170, 130, 95))
    if style != 'line' and RNG.random() < 0.3:  # planta
        c = (int(x0 + m + rnd(0, w * 0.8)), int(y0 + m + rnd(0, h * 0.8)))
        for _ in range(6):
            cv2.circle(img, (c[0] + int(rnd(-8, 8) * SS), c[1] + int(rnd(-8, 8) * SS)),
                       int(rnd(4, 9) * SS), (60, int(rnd(110, 150)), 60), -1)


def extras(img, rect, kind, px, style):
    """v4: objetos que el modelo v3 confundía — todos cuentan como ESPACIO en la máscara.
    Islas y mesones oscuros, closets oscuros, alfombras claras grandes, sofás oscuros,
    colchas con textura y líneas rojas punteadas de mobiliario."""
    if style == 'line':
        return
    x0, y0, x1, y1 = [float(v) for v in rect]
    w, h = x1 - x0, y1 - y0
    if w < 0.9 * px or h < 0.9 * px:
        return

    def dark():
        g = rnd(15, 85)
        return (g + rnd(0, 12), g + rnd(0, 10), g + rnd(0, 18))

    def rect_in(dw, dh, touch=False):
        """Rectángulo de dw x dh metros dentro del espacio; touch=True lo pega a un muro."""
        rw, rh = min(dw * px, w - 4), min(dh * px, h - 4)
        if RNG.random() < 0.5:
            rw, rh = rh, rw
        rw, rh = min(rw, w - 4), min(rh, h - 4)
        if touch:
            side = RNG.integers(4)
            a = x0 + 2 if side == 0 else (x1 - rw - 2 if side == 1 else rnd(x0 + 2, max(x0 + 3, x1 - rw - 2)))
            b = y0 + 2 if side == 2 else (y1 - rh - 2 if side == 3 else rnd(y0 + 2, max(y0 + 3, y1 - rh - 2)))
        else:
            a = rnd(x0 + 0.4 * px, max(x0 + 0.4 * px + 1, x1 - rw - 0.4 * px))
            b = rnd(y0 + 0.4 * px, max(y0 + 0.4 * px + 1, y1 - rh - 0.4 * px))
        return int(a), int(b), int(a + rw), int(b + rh)

    def shade(a, b, c, d):
        cv2.rectangle(img, (a + 2 * SS, b + 2 * SS), (c + 2 * SS, d + 2 * SS),
                      tuple(float(v) * 0.7 for v in img[min(b, img.shape[0] - 1), min(a, img.shape[1] - 1)]), -1)

    # alfombra clara grande (antes de los muebles que van encima)
    if kind in ('living', 'living2', 'bed') and RNG.random() < 0.55:
        a, b, c, d = rect_in(rnd(1.6, 3.2), rnd(1.4, 2.4))
        g = rnd(215, 252)
        cv2.rectangle(img, (a, b), (c, d), (g, g, g - rnd(0, 12)), -1)
        if RNG.random() < 0.5:  # textura / flecos
            for k in range(a, c, max(2, SS * 3)):
                cv2.line(img, (k, b), (k, d), (g - 8, g - 8, g - 14), 1)
    # isla o mesón oscuro
    if kind in ('kitchen', 'living', 'living2') and RNG.random() < 0.6:
        a, b, c, d = rect_in(rnd(1.2, 3.0), rnd(0.55, 1.0), touch=RNG.random() < 0.5)
        shade(a, b, c, d)
        cv2.rectangle(img, (a, b), (c, d), dark(), -1)
        orect(a, b, c, d, MESON)
        if RNG.random() < 0.5:  # estufa o lavaplatos sobre el mesón
            s = int(0.25 * px)
            cx, cy = (a + c) // 2, (b + d) // 2
            cv2.rectangle(img, (cx - s, cy - s // 2), (cx + s, cy + s // 2), (rnd(120, 200),) * 3, -1)
            orect(cx - s, cy - s // 2, cx + s, cy + s // 2, APARATO)
        if RNG.random() < 0.6:  # banquitos
            for k in range(3):
                cv2.circle(img, (int(a + (k + 0.5) * (c - a) / 3), d + int(0.25 * px)), int(0.15 * px),
                           (rnd(60, 200),) * 3, -1)
                ocircle((a + (k + 0.5) * (c - a) / 3, d + 0.25 * px), 0.15 * px, MUEBLE)
    # closet oscuro pegado al muro
    if kind in ('bed', 'study', 'closet') and RNG.random() < 0.6:
        a, b, c, d = rect_in(rnd(1.2, 2.6), 0.6, touch=True)
        shade(a, b, c, d)
        col = dark() if RNG.random() < 0.6 else (rnd(70, 120), rnd(90, 140), rnd(120, 170))
        cv2.rectangle(img, (a, b), (c, d), col, -1)
        orect(a, b, c, d, MESON)
        for k in range(1, 4):  # puertas del closet
            if c - a > d - b:
                xx = a + k * (c - a) // 4; cv2.line(img, (xx, b), (xx, d), (100, 100, 100), 1)
            else:
                yy = b + k * (d - b) // 4; cv2.line(img, (a, yy), (c, yy), (100, 100, 100), 1)
    # sofá oscuro
    if kind in ('living', 'living2') and RNG.random() < 0.35:
        a, b, c, d = rect_in(rnd(1.8, 2.6), rnd(0.8, 0.95), touch=RNG.random() < 0.4)
        shade(a, b, c, d)
        cv2.rectangle(img, (a, b), (c, d), dark(), -1)
        orect(a, b, c, d, MUEBLE)
        cv2.rectangle(img, (a + int(0.15 * px), b + int(0.15 * px)), (c - int(0.15 * px), d - int(0.15 * px)),
                      tuple(min(255, v + 25) for v in dark()), -1)
    # colcha con rayas / textura
    if kind == 'bed' and RNG.random() < 0.5:
        a, b, c, d = rect_in(1.5, 2.0)
        base = (rnd(90, 230), rnd(90, 230), rnd(90, 230))
        cv2.rectangle(img, (a, b), (c, d), base, -1)
        orect(a, b, c, d, MUEBLE)
        step = max(3, int(rnd(0.05, 0.15) * px))
        for k in range(b, d, step):
            cv2.line(img, (a, k), (c, k), tuple(v * 0.8 for v in base), max(1, SS))
    # aparatos fijos de cocina, ropas y baño
    if kind == 'kitchen' and RNG.random() < 0.7:  # nevera
        a, b, c, d = rect_in(0.7, 0.7, touch=True)
        g = rnd(170, 245)
        cv2.rectangle(img, (a, b), (c, d), (g, g, g), -1); cv2.rectangle(img, (a, b), (c, d), (90, 90, 90), SS)
        orect(a, b, c, d, APARATO)
    if kind in ('laundry', 'kitchen') and RNG.random() < 0.6:  # lavadora / secadora
        a, b, c, d = rect_in(0.6, 0.6, touch=True)
        cv2.rectangle(img, (a, b), (c, d), (235, 235, 235), -1); cv2.rectangle(img, (a, b), (c, d), (90, 90, 90), SS)
        cv2.circle(img, ((a + c) // 2, (b + d) // 2), max(2, (c - a) // 3), (120, 120, 120), SS)
        orect(a, b, c, d, APARATO)
    if kind == 'bath':
        if RNG.random() < 0.7:  # ducha o tina
            tub = RNG.random() < 0.3
            a, b, c, d = rect_in(1.6 if tub else 0.9, 0.75 if tub else 0.9, touch=True)
            cv2.rectangle(img, (a, b), (c, d), (rnd(220, 250),) * 3, -1)
            cv2.rectangle(img, (a, b), (c, d), (110, 110, 110), SS)
            if tub:
                cv2.ellipse(img, ((a + c) // 2, (b + d) // 2), (max(2, (c - a) // 2 - 4), max(2, (d - b) // 2 - 4)),
                            0, 0, 360, (150, 150, 150), SS)
            else:
                cv2.line(img, (a, b), (c, d), (160, 160, 160), 1); cv2.line(img, (c, b), (a, d), (160, 160, 160), 1)
                cv2.circle(img, ((a + c) // 2, (b + d) // 2), max(2, int(0.04 * px)), (90, 90, 90), -1)
            orect(a, b, c, d, APARATO)
        if RNG.random() < 0.5:  # mueble de lavamanos (mesón) con lavamanos (aparato)
            a, b, c, d = rect_in(rnd(0.8, 1.4), 0.5, touch=True)
            cv2.rectangle(img, (a, b), (c, d), (rnd(80, 230),) * 3, -1)
            orect(a, b, c, d, MESON)
            cx, cy = (a + c) // 2, (b + d) // 2
            r = max(2, int(0.17 * px))
            cv2.ellipse(img, (cx, cy), (r, int(r * 0.75)), 0, 0, 360, (250, 250, 250), -1)
            oellipse((cx, cy), (r, r * 0.75), APARATO)
    # líneas rojas punteadas de mobiliario fijo (convención de algunos folletos)
    if RNG.random() < 0.25:
        a, b, c, d = rect_in(rnd(0.6, 2.0), rnd(0.4, 0.7), touch=True)
        for (p, q) in (((a, b), (c, b)), ((c, b), (c, d)), ((c, d), (a, d)), ((a, d), (a, b))):
            n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / (4 * SS)))
            for k in range(0, n, 2):
                t0, t1 = k / n, min(1, (k + 1) / n)
                cv2.line(img, (int(p[0] + (q[0] - p[0]) * t0), int(p[1] + (q[1] - p[1]) * t0)),
                         (int(p[0] + (q[0] - p[0]) * t1), int(p[1] + (q[1] - p[1]) * t1)), (40, 40, 210), SS)


# ---------------------------------------------------------------- render
def render(size=320, style=None):
    style = style or RNG.choice(['co', 'co', 'us', 'line'])
    L = make_layout()
    S = size * SS
    margin = rnd(0.06, 0.22)
    px = (S * (1 - 2 * margin)) / max(L['W'], L['H'])  # píxeles por metro (supermuestreado)
    global LAST
    LAST = dict(layout=L, px_per_m=px / SS)   # para calcular el área real de la planta generada
    ox = (S - L['W'] * px) / 2 + rnd(-0.05, 0.05) * S
    oy = (S - L['H'] * px) / 2 + rnd(-0.05, 0.05) * S

    def P(x, y):
        return int(round(ox + x * px)), int(round(oy + y * px))

    img = background(S, S, style)
    mask = np.zeros((S, S), np.uint8)
    global OBJ
    OBJ = np.zeros((S, S), np.uint8)

    t_ext = rnd(0.15, 0.30) * px
    t_int = rnd(0.07, 0.15) * px
    LAST.update(ox=ox / SS, oy=oy / SS, t_ext_px=max(3, int(t_ext)) / SS, t_int_px=2 * max(2, int(t_int / 2)) / SS, rot=0)
    _b = rnd(105, 185); _g = _b + rnd(8, 25)
    wood = [(_b, _g, min(250, _g + rnd(8, 28)))]  # BGR: madera gris-café
    # pisos
    for r, t in zip(L['rooms'], L['types']):
        a, b = P(r[0], r[1])
        c, d = P(r[2], r[3])
        h, w = d - b, c - a
        if h <= 0 or w <= 0:
            continue
        if style == 'line':
            patch = np.full((h, w, 3), 255, np.float32)
        elif t in ('bath', 'laundry', 'kitchen') and RNG.random() < 0.8:
            g = rnd(200, 235)
            patch = tex_tile(h, w, (g, g, g + rnd(-5, 5)))
        elif style == 'us' or RNG.random() < 0.25:
            g = rnd(185, 225)
            patch = tex_flat(h, w, (g, g, g)) if RNG.random() < 0.5 else tex_wood(h, w, wood[0])
        else:
            patch = tex_wood(h, w, wood[0])
        img[b:d, a:c] = patch
        mask[b:d, a:c] = 1
    # muebles
    for r, t in zip(L['rooms'], L['types']):
        a, b = P(r[0], r[1])
        c, d = P(r[2], r[3])
        if RNG.random() < 0.9:
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
            if not on_border or RNG.random() < 0.3:
                continue
            if side in (0, 2):
                span = r[2] - r[0]
                ln = min(span - 0.6, rnd(0.8, 2.6))
                if ln < 0.6:
                    continue
                s0 = r[0] + rnd(0.3, span - ln - 0.3)
                y = r[1] if side == 0 else r[3]
                p0, p1 = P(s0, y), P(s0 + ln, y)
                cv2.rectangle(win_m, (p0[0], p0[1] - k // 2 - 1), (p1[0], p1[1] + k // 2 + 1), 255, -1)
            else:
                span = r[3] - r[1]
                ln = min(span - 0.6, rnd(0.8, 2.6))
                if ln < 0.6:
                    continue
                s0 = r[1] + rnd(0.3, span - ln - 0.3)
                x = r[2] if side == 1 else r[0]
                p0, p1 = P(x, s0), P(x, s0 + ln)
                cv2.rectangle(win_m, (p0[0] - k // 2 - 1, p0[1]), (p1[0] + k // 2 + 1, p1[1]), 255, -1)
    win_m &= wall_m

    # puertas interiores
    door_draw = []
    for (o, f, lo, hi) in L['doors']:
        dw = min(hi - lo - 0.2, rnd(0.7, 0.95))
        s0 = rnd(lo + 0.1, hi - dw - 0.1)
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
        r = L['rooms'][RNG.integers(len(L['rooms']))]
        if r[1] < 1e-6 and r[2] - r[0] > 1.4:
            s0 = rnd(r[0] + 0.2, r[2] - 1.1)
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
        wc = rnd(0, 40)
        img[solid] = wc
    elif style == 'us':
        sh = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 3 * SS)
        img *= (1 - 0.35 * np.roll(np.roll(sh, 3 * SS, 0), 3 * SS, 1))[..., None]
        wc = rnd(225, 250)
        img[solid] = wc
        edges = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[edges] = rnd(120, 170)
    else:
        img[solid] = rnd(0, 60) if RNG.random() < 0.5 else 255
        edges = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[edges] = 20
    # ventanas: varias convenciones gráficas
    wm = win_m > 0
    wv = RNG.choice(['thin', 'glass', 'gray', 'gap'])
    wcol = (rnd(150, 210), rnd(150, 200), rnd(140, 190))
    if style == 'line':
        wv = RNG.choice(['thin', 'gap'])
    if wv == 'glass':  # vidrio azul claro con marco
        img[wm] = (rnd(220, 250), rnd(200, 230), rnd(160, 200))
    elif wv == 'gray':  # banda gris clara
        img[wm] = rnd(170, 215)
    else:
        img[wm] = (250, 250, 250) if style != 'line' else (255, 255, 255)
    we = cv2.morphologyEx(win_m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if wv != 'gap':
        img[we & wm] = wcol
    if wv == 'thin':  # línea central (doble línea)
        img[cv2.erode(win_m, np.ones((max(3, k // 2), max(3, k // 2)), np.uint8)) > 0] = \
            np.array(wcol) * 0.9 if RNG.random() < 0.6 else (230, 230, 230)
    # puertas: hueco + hoja + arco
    for (o, p0, p1) in door_draw:
        if RNG.random() < 0.65:
            r_ = abs((p1[0] - p0[0]) if o == 'h' else (p1[1] - p0[1]))
            sgn = 1 if RNG.random() < 0.5 else -1
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
    if ent and style != 'line' and RNG.random() < 0.7:
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

    for _ in range(RNG.integers(0, 7)):
        kind = RNG.choice(['title', 'small', 'logo', 'area'])
        x, y = int(rnd(0, S * 0.85)), int(rnd(S * 0.03, S * 0.97))
        if kind == 'logo':
            s_ = int(rnd(0.07, 0.16) * S)
            if free(x, y, x + s_, y + s_):
                col = tuple(rnd(40, 200) for _ in range(3))
                cv2.rectangle(img, (x, y), (x + s_, y + s_), col, -1)
                cv2.putText(img, ''.join(RNG.choice(list('AVELNRDOT'), 4)), (x + s_ // 8, y + s_ // 2),
                            fonts[RNG.integers(4)], s_ / 90, (255, 255, 255), SS, cv2.LINE_AA)
            continue
        txt = {'title': 'APTO TIPO ' + RNG.choice(list('ABCDE')),
               'area': f'{RNG.integers(35, 140)}m2',
               'small': ''.join(RNG.choice(list('ABCDEFGHIJKLMNOPRSTUV '), RNG.integers(8, 40)))}[kind]
        sc = {'title': rnd(0.7, 1.4), 'area': rnd(0.6, 1.2), 'small': rnd(0.25, 0.45)}[kind] * SS
        (tw, th), _ = cv2.getTextSize(txt, fonts[0], sc, SS)
        if free(x, y - th, x + tw, y + 4):
            cv2.putText(img, txt, (x, y), fonts[RNG.integers(4)], sc, (rnd(0, 90),) * 3,
                        max(1, int(SS * rnd(0.6, 1.5))), cv2.LINE_AA)

    # máscara final
    obj_in = (mask == 1) & (OBJ > 0)
    mask[obj_in] = OBJ[obj_in]
    mask[wall_m > 0] = 2
    mask[(door_m > 0)] = 3
    mask[(win_m > 0)] = 4
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    # degradaciones
    if RNG.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rnd(0.3, 1.0))
    if RNG.random() < 0.6:
        q = int(rnd(40, 95))
        img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    if RNG.random() < 0.5:
        img = np.clip(img.astype(np.float32) * rnd(0.85, 1.15) + rnd(-15, 15), 0, 255).astype(np.uint8)
    if RNG.random() < 0.5:  # rotación de 90°
        kk = int(RNG.integers(1, 4))
        img, mask = np.rot90(img, kk).copy(), np.rot90(mask, kk).copy()
        LAST['rot'] = kk
    return img, mask, style


PALETTE = np.array([[255, 255, 255], [200, 230, 255], [30, 30, 30], [0, 140, 255], [255, 80, 40],
                    [40, 160, 40], [200, 60, 200], [150, 210, 160]], np.uint8)  # +mesón, aparato, mueble


def colorize(mask):
    return PALETTE[mask]


def net_area(mask):
    """Área interior neta (m²) de la última planta generada: espacios + objetos, sin muros."""
    px = LAST['px_per_m']
    return float(np.isin(mask, (1, 5, 6, 7)).sum()) / px / px
