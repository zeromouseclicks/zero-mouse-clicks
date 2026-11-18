"""Bloques paramétricos de muebles y aparatos (v0.3), dibujados como en una oficina de arquitectura.

Cada bloque se dibuja en coordenadas locales en METROS: x a lo ancho (-w/2..w/2), y en profundidad con el
respaldo arriba (y = +d/2) y el frente abajo (y = -d/2). draw() hace la traslación, el giro y la escala.

Dos estilos:
- 'line'  láminas técnicas: trazo negro/gris, relleno blanco (planta A-101, cortes)
- 'color' lámina de mobiliario A-105: rellenos suaves tipo folleto y sombra
"""
import math

from reportlab.lib.colors import Color

# trazos en puntos (se dividen por la escala al dibujar en metros)
LW_OUT, LW_DET = 0.4, 0.18

PAL = {   # relleno en estilo 'color'
    'wood': (0.86, 0.76, 0.63), 'wood_dk': (0.62, 0.48, 0.36), 'fabric': (0.80, 0.83, 0.86),
    'fabric_dk': (0.66, 0.70, 0.74), 'linen': (0.97, 0.96, 0.94), 'cover': (0.86, 0.89, 0.92),
    'white': (1, 1, 1), 'steel': (0.84, 0.85, 0.87), 'dark': (0.30, 0.31, 0.33), 'leaf': (0.55, 0.72, 0.47),
    'leaf_dk': (0.36, 0.55, 0.32), 'ceramic': (0.98, 0.98, 0.99), 'cushion': (0.90, 0.80, 0.68),
}


class Pen:
    def __init__(self, c, k, style):
        self.c, self.k, self.style = c, k, style

    def lw(self, pts):
        self.c.setLineWidth(pts / self.k)

    def fill(self, name):
        if self.style == 'color':
            r, g, b = PAL.get(name, PAL['white'])
            self.c.setFillColorRGB(r, g, b)
        else:
            self.c.setFillColorRGB(1, 1, 1)

    def stroke(self, gray=None):
        if self.style == 'color':
            self.c.setStrokeColorRGB(0.28, 0.27, 0.26)
        else:
            self.c.setStrokeGray(gray if gray is not None else 0)

    def rect(self, x0, y0, x1, y1, fill=None, r=0.0, lw=LW_OUT):
        self.lw(lw)
        if fill:
            self.fill(fill)
        if r > 0:
            self.c.roundRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0), r, stroke=1, fill=1 if fill else 0)
        else:
            self.c.rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0), stroke=1, fill=1 if fill else 0)

    def circle(self, x, y, r, fill=None, lw=LW_OUT):
        self.lw(lw)
        if fill:
            self.fill(fill)
        self.c.circle(x, y, r, stroke=1, fill=1 if fill else 0)

    def ellipse(self, x0, y0, x1, y1, fill=None, lw=LW_OUT):
        self.lw(lw)
        if fill:
            self.fill(fill)
        self.c.ellipse(x0, y0, x1, y1, stroke=1, fill=1 if fill else 0)

    def line(self, x0, y0, x1, y1, lw=LW_DET):
        self.lw(lw)
        self.c.line(x0, y0, x1, y1)


# ------------------------------------------------------------------------------------------- bloques
def bed(p, w, d, it):
    hw, hd = w / 2, d / 2
    p.rect(-hw, -hd, hw, hd, 'wood')
    p.rect(-hw, hd - 0.07, hw, hd, 'wood_dk', lw=LW_DET)                     # cabecera
    p.rect(-hw + 0.03, -hd + 0.03, hw - 0.03, hd - 0.10, 'linen', lw=LW_DET)  # colchón / sábana
    n = 2 if w >= 1.2 else 1
    gap = 0.06
    pw = (w - 0.16 - gap * (n - 1)) / n
    for i in range(n):
        x0 = -hw + 0.08 + i * (pw + gap)
        p.rect(x0, hd - 0.42, x0 + pw, hd - 0.13, 'white', r=0.05, lw=LW_DET)
    top = hd - 0.55                                                             # cobija con doblez
    p.rect(-hw + 0.02, -hd + 0.02, hw - 0.02, top, 'cover', lw=LW_DET)
    p.rect(-hw + 0.02, top - 0.22, hw - 0.02, top, 'linen', lw=LW_DET)
    f = min(0.38, w * 0.3)
    p.c.setFillColorRGB(*((0.97, 0.96, 0.94) if p.style == 'color' else (1, 1, 1)))
    path = p.c.beginPath(); path.moveTo(hw - 0.02, -hd + 0.02 + f); path.lineTo(hw - 0.02 - f, -hd + 0.02)
    path.lineTo(hw - 0.02, -hd + 0.02); path.close(); p.lw(LW_DET); p.c.drawPath(path, stroke=1, fill=1)


def nightstand(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'wood')
    r = min(w, d) * 0.22
    p.circle(0, 0.03, r, 'white', lw=LW_DET)
    p.line(-r * 0.7, 0.03 - r * 0.7, r * 0.7, 0.03 + r * 0.7); p.line(-r * 0.7, 0.03 + r * 0.7, r * 0.7, 0.03 - r * 0.7)


def desk(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'wood')
    lw_ = min(0.34, w * 0.3)
    p.rect(-lw_ / 2, d / 2 - 0.30, lw_ / 2, d / 2 - 0.08, 'steel', lw=LW_DET)   # portátil / monitor


def chair(p, w, d, it):
    p.rect(-w / 2 + 0.02, -d / 2, w / 2 - 0.02, d / 2 - 0.06, 'fabric', r=0.05)
    p.rect(-w / 2, d / 2 - 0.08, w / 2, d / 2, 'wood_dk', r=0.03, lw=LW_DET)


def armchair(p, w, d, it):
    hw, hd = w / 2, d / 2
    arm = 0.14 * w / 0.85
    p.rect(-hw, -hd, hw, hd, 'fabric', r=0.08)
    p.rect(-hw + arm, -hd + 0.03, hw - arm, hd - 0.20, 'fabric_dk' if p.style == 'color' else None, r=0.05, lw=LW_DET)
    p.line(-hw + 0.02, hd - 0.18, hw - 0.02, hd - 0.18)


def sofa(p, w, d, it):
    hw, hd = w / 2, d / 2
    arm, back = 0.18, 0.20
    p.rect(-hw, -hd, hw, hd, 'fabric', r=0.08)
    p.line(-hw + arm, hd - back, hw - arm, hd - back)
    p.line(-hw + arm, -hd, -hw + arm, hd - back); p.line(hw - arm, -hd, hw - arm, hd - back)
    n = max(2, int(round((w - 2 * arm) / 0.62)))
    sw = (w - 2 * arm) / n
    for i in range(n):
        x0 = -hw + arm + i * sw
        p.rect(x0 + 0.02, -hd + 0.04, x0 + sw - 0.02, hd - back - 0.02, 'fabric_dk' if p.style == 'color' else None, r=0.04, lw=LW_DET)
    if p.style == 'color':                                                     # cojines de respaldo
        for x in (-hw + arm + 0.25, hw - arm - 0.25):
            p.rect(x - 0.18, hd - back - 0.10, x + 0.18, hd - back + 0.02, 'cushion', r=0.04, lw=LW_DET)


def coffee_table(p, w, d, it):
    if it.get('round'):
        p.circle(0, 0, w / 2, 'wood'); p.circle(0, 0, w / 2 - 0.05, None, lw=LW_DET)
    else:
        p.rect(-w / 2, -d / 2, w / 2, d / 2, 'wood', r=0.03)
        p.rect(-w / 2 + 0.05, -d / 2 + 0.05, w / 2 - 0.05, d / 2 - 0.05, None, lw=LW_DET)


def side_table(p, w, d, it):
    p.circle(0, 0, w / 2, 'wood')


def tv_unit(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'wood')
    tw = min(1.4, w * 0.75)
    p.rect(-tw / 2, d / 2 - 0.16, tw / 2, d / 2 - 0.10, 'dark', lw=LW_DET)
    p.rect(-0.12, d / 2 - 0.22, 0.12, d / 2 - 0.16, 'dark', lw=LW_DET)


def dining_table(p, w, d, it):
    if it.get('round'):
        p.circle(0, 0, w / 2, 'wood'); p.circle(0, 0, 0.10, 'leaf' if p.style == 'color' else None, lw=LW_DET)
    else:
        p.rect(-w / 2, -d / 2, w / 2, d / 2, 'wood', r=0.02)
        p.line(-w / 2 + 0.06, -d / 2 + 0.06, w / 2 - 0.06, -d / 2 + 0.06)


def stool(p, w, d, it):
    p.circle(0, 0, w / 2, 'fabric'); p.circle(0, 0, w / 2 - 0.06, None, lw=LW_DET)


def plant(p, w, d, it):
    r = w / 2
    p.circle(0, 0, r * 0.45, 'wood_dk', lw=LW_DET)
    for k in range(7):
        a = 2 * math.pi * k / 7 + 0.3
        cx, cy = 0.55 * r * math.cos(a), 0.55 * r * math.sin(a)
        p.c.saveState(); p.c.translate(cx, cy); p.c.rotate(math.degrees(a))
        p.ellipse(-0.45 * r, -0.2 * r, 0.45 * r, 0.2 * r, 'leaf', lw=LW_DET)
        p.c.restoreState()
    p.circle(0, 0, r * 0.18, 'leaf_dk', lw=LW_DET)


def misc(p, w, d, it):
    """Algo hay ahí pero no se sabe qué: contorno a trazos, sin relleno de mueble."""
    p.c.setDash(0.06, 0.04)
    p.rect(-w / 2, -d / 2, w / 2, d / 2, None, r=0.04, lw=LW_DET)
    p.c.setDash()


def fridge(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'steel')
    p.line(-w / 2, -d / 2 + 0.05, w / 2, -d / 2 + 0.05)                           # puerta
    p.line(-w / 2, d / 2, w / 2, -d / 2 + 0.05); p.line(w / 2, d / 2, -w / 2, -d / 2 + 0.05)


def cooktop(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'dark' if p.style == 'color' else None)
    if p.style == 'color':
        p.c.setStrokeColorRGB(0.85, 0.85, 0.85)
    for (x, y, r) in ((-0.14, 0.10, 0.09), (0.14, 0.10, 0.07), (-0.14, -0.12, 0.07), (0.14, -0.12, 0.09)):
        sx, sy = w / 0.60, d / 0.52
        p.circle(x * sx, y * sy, r * min(sx, sy), None, lw=LW_DET)
        p.circle(x * sx, y * sy, r * min(sx, sy) * 0.45, None, lw=LW_DET)
    p.stroke()


def sink(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'steel', lw=LW_DET)
    n = 2 if w >= 0.75 else 1
    bw = (w - 0.10 - 0.04 * (n - 1)) / n
    for i in range(n):
        x0 = -w / 2 + 0.05 + i * (bw + 0.04)
        p.rect(x0, -d / 2 + 0.05, x0 + bw, d / 2 - 0.10, 'white', r=0.04, lw=LW_DET)
        p.circle(x0 + bw / 2, -d / 2 + 0.05 + (d - 0.15) / 2, 0.02, None, lw=LW_DET)
    p.circle(0, d / 2 - 0.05, 0.025, 'steel', lw=LW_DET)


def washer(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'white')
    p.line(-w / 2, d / 2 - 0.10, w / 2, d / 2 - 0.10)
    p.circle(0, -0.05, min(w, d) * 0.30, 'steel', lw=LW_DET)
    p.circle(0, -0.05, min(w, d) * 0.20, None, lw=LW_DET)


def toilet(p, w, d, it):
    hw, hd = w / 2, d / 2
    p.rect(-hw, hd - 0.20, hw, hd, 'ceramic', r=0.03)                          # tanque
    p.ellipse(-hw * 0.88, -hd, hw * 0.88, hd - 0.16, 'ceramic')                # taza
    p.ellipse(-hw * 0.6, -hd + 0.07, hw * 0.6, hd - 0.26, None, lw=LW_DET)     # asiento


def lavatory(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'ceramic')
    n = 2 if w >= 1.15 else 1
    for i in range(n):
        cx = 0 if n == 1 else (-w / 4 if i == 0 else w / 4)
        bw = min(0.46, (w - 0.1) / n - 0.06)
        p.ellipse(cx - bw / 2, -d / 2 + 0.06, cx + bw / 2, d / 2 - 0.10, 'white', lw=LW_DET)
        p.circle(cx, d / 2 - 0.05, 0.02, 'steel', lw=LW_DET)


def shower(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'ceramic', lw=LW_DET)
    p.line(-w / 2, -d / 2, w / 2, d / 2); p.line(-w / 2, d / 2, w / 2, -d / 2)
    p.circle(0, 0, 0.04, 'steel', lw=LW_DET)


def bathtub(p, w, d, it):
    p.rect(-w / 2, -d / 2, w / 2, d / 2, 'ceramic')
    p.rect(-w / 2 + 0.07, -d / 2 + 0.07, w / 2 - 0.07, d / 2 - 0.07, 'white', r=0.15, lw=LW_DET)
    p.circle(-w / 2 + 0.22, 0, 0.03, None, lw=LW_DET)


BLOCKS = dict(bed=bed, nightstand=nightstand, desk=desk, chair=chair, armchair=armchair, sofa=sofa,
              coffee_table=coffee_table, side_table=side_table, tv_unit=tv_unit, dining_table=dining_table,
              stool=stool, plant=plant, misc=misc, fridge=fridge, cooktop=cooktop, sink=sink, washer=washer,
              toilet=toilet, lavatory=lavatory, shower=shower, bathtub=bathtub)
ROUND = ('stool', 'plant', 'side_table')


def footprint(c, w, d, it):
    if it['type'] in ROUND or (it['type'] in ('coffee_table', 'dining_table') and it.get('round')):
        c.circle(0, 0, w / 2, stroke=0, fill=1)
    else:
        c.rect(-w / 2, -d / 2, w, d, stroke=0, fill=1)


def draw(c, it, X, Y, k, style='line', gray=0.0, shadow=None):
    """Dibuja la pieza `it` con centro en (X, Y) puntos de la lámina; k = puntos por metro."""
    fn = BLOCKS.get(it['type'], misc)
    w, d = it['w'], it['d']
    if style == 'color' and shadow is not None and it['type'] not in ('plant', 'misc'):
        c.saveState()
        c.translate(X + shadow * k, Y - shadow * k); c.rotate(it['ang']); c.scale(k, k)
        c.setFillColor(Color(0.2, 0.2, 0.2, alpha=0.16))
        footprint(c, w, d, it)
        c.restoreState()
    c.saveState()
    c.translate(X, Y); c.rotate(it['ang']); c.scale(k, k)
    p = Pen(c, k, style)
    p.stroke(gray)
    c.setLineJoin(1)
    fn(p, w, d, it)
    c.restoreState()
