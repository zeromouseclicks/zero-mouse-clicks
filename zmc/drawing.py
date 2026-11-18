"""Modelo vectorial -> juego de planos PDF (A3 horizontal), solo reglas.

Lámina A-101: planta con ejes, cotas en cadena y totales, vanos, rótulos de espacio,
escala gráfica, rótulo y nota obligatoria.
Lámina A-102: cuadro de áreas, cuadro de puertas y ventanas, supuestos.
"""
import datetime
import math
import numpy as np
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from . import layout as L
from .layout import F, FB, PAGE_W, PAGE_H, FRAME, TB_W, NOTE, ACC

SCALES = [20, 25, 50, 75, 100, 125, 150, 200]


def _cluster(vals, tol):
    vals = sorted(vals)
    out, cur = [], [vals[0]] if vals else []
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            out.append(float(np.mean(cur))); cur = [v]
    if cur:
        out.append(float(np.mean(cur)))
    return out


class Plan:
    """Convierte el modelo en metros (origen abajo-izquierda de la huella)."""

    def __init__(self, model, fin=None):
        self.m = model
        self.fin = fin
        s = model['m_per_px']
        x0, y0, x1, y1 = model['bbox']
        self.s, self.x0, self.y1 = s, x0, y1
        self.W, self.H = (x1 - x0) * s, (y1 - y0) * s
        self.walls = [(o, self.X(a), self.Y(d), self.X(c), self.Y(b)) for o, a, b, c, d in model['walls']]
        self.ops = []
        for k, op in enumerate(sorted(model['openings'], key=lambda o: (o['kind'], o['y0'], o['x0']))):
            r = dict(op)
            r.update(X0=self.X(op['x0']), Y0=self.Y(op['y1']), X1=self.X(op['x1']), Y1=self.Y(op['y0']))
            r['width'] = (r['X1'] - r['X0']) if op['orient'] == 'h' else (r['Y1'] - r['Y0'])
            self.ops.append(r)
        # vanos imposibles a escala real: se descartan
        self.ops = [o for o in self.ops if o['width'] >= (0.55 if o['kind'] == 'door' else 0.45)
                    and o['width'] <= (2.8 if o['kind'] == 'door' else 8.5)]
        d = [o for o in self.ops if o['kind'] == 'door']
        w = [o for o in self.ops if o['kind'] == 'window']
        for i, o in enumerate(d):
            o['tag'] = f'D-{i + 1:02d}'
        for i, o in enumerate(w):
            o['tag'] = f'W-{i + 1:02d}'
        net = model.get('net_px') or {}
        self.rooms = [dict(id=r['id'], area=net.get(r['id'], r['area_px']) * s * s, X=self.X(r['cx']), Y=self.Y(r['cy']))
                      for r in model['rooms']]
        self.t = model['thickness_px'] * s
        self.objs, self.elec, self.contours, self.items = [], [], {}, []
        if fin:
            import cv2
            for o in fin['objects']:
                q = dict(o); q.update(X0=self.X(o['x0']), Y0=self.Y(o['y1']), X1=self.X(o['x1']), Y1=self.Y(o['y0']))
                self.objs.append(q)
            self.elec = [dict(p, X=self.X(p['x']), Y=self.Y(p['y'])) for p in fin['electrical']]
            self.items = [dict(it, X=self.X(it['cx']), Y=self.Y(it['cy'])) for it in fin.get('items', [])]
            lab = model['room_labels']
            for r in self.rooms:
                info = fin['rooms'].get(r['id'], {})
                r['zones'] = [(n, self.X(zx), self.Y(zy)) for n, zx, zy in info.get('zones', [])]
                r['exterior'] = info.get('type') == 'patio'
                r.update(name=info.get('name', f"S-{r['id']:02d}"), type=info.get('type', ''),
                         floor=info.get('floor', ''), floor_dir=info.get('floor_dir', 'h'), open_kitchen=info.get('open_kitchen', False))
                cs, _ = cv2.findContours((lab == info.get('label', -1)).astype('uint8'), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True)[:, 0, :]
                    self.contours[r['id']] = [(self.X(px), self.Y(py)) for px, py in cnt]
        # ejes: líneas centrales de muros largos
        vx = [(a + c) / 2 for o, a, b, c, d in self.walls if (d - b) > 1.8 and (c - a) < 3 * self.t]
        hy = [(b + d) / 2 for o, a, b, c, d in self.walls if (c - a) > 1.8 and (d - b) < 3 * self.t]
        self.ax_x = _cluster(vx, 0.6) if vx else [0, self.W]
        self.ax_y = _cluster(hy, 0.6) if hy else [0, self.H]

    def X(self, px):
        return (px - self.x0) * self.s

    def Y(self, py):
        return (self.y1 - py) * self.s


def _frame(c, title, number, meta, scale_txt):
    """Marco ISO 5457 con zonas y columna de rótulo (layout.py). Devuelve la x de la columna."""
    return L.frame(c, title, number, meta, scale_txt)


def plan_layout(P, comp=True):
    """Escala normalizada más grande que cabe y sitio para un panel de datos al lado o debajo."""
    dx0, dy0, dx1, dy1 = L.drawing_area()
    DW, DH = dx1 - dx0, dy1 - dy0
    left, right, top, bottom, pad = 1.75, 1.0, 1.75, 1.0, 8 * mm
    for sc in SCALES:
        k = 1000 / sc * mm
        wpt = (P.W + left + right) * k + 2 * pad
        hpt = (P.H + top + bottom) * k + 2 * pad
        if wpt <= DW and hpt <= DH:
            break
    lw, lh = DW - wpt, DH - hpt
    out = dict(scale=sc, k=k, comp=None)
    if comp and lw >= 58 * mm:
        cw = min(lw - 6 * mm, 96 * mm)
        out['comp'] = (dx1 - cw, dy0 + 2 * mm, cw, DH - 4 * mm)
        out['ox'] = dx0 + pad + left * k + max(0, (lw - cw - 6 * mm) / 2)
        out['oy'] = dy0 + pad + bottom * k + lh / 2
    else:
        out['ox'] = dx0 + pad + left * k + lw / 2
        out['oy'] = dy0 + pad + bottom * k + lh / 2
    return out


def _dim(c, p0, p1, off, txt, horizontal, k):
    """Cota con líneas de referencia y marcas oblicuas. p en puntos de página."""
    c.setLineWidth(0.25)
    if horizontal:
        y = off
        c.line(p0[0], p0[1], p0[0], y - 1.5 * mm * np.sign(y - p0[1]))
        c.line(p1[0], p1[1], p1[0], y - 1.5 * mm * np.sign(y - p1[1]))
        c.line(p0[0] - 1.5 * mm, y, p1[0] + 1.5 * mm, y)
        for x in (p0[0], p1[0]):
            c.setLineWidth(0.5); c.line(x - 1 * mm, y - 1 * mm, x + 1 * mm, y + 1 * mm); c.setLineWidth(0.25)
        if p1[0] - p0[0] > 7 * mm:
            c.setFont(F, 6.5); c.drawCentredString((p0[0] + p1[0]) / 2, y + 1 * mm, txt)
    else:
        x = off
        c.line(p0[0], p0[1], x + 1.5 * mm * np.sign(p0[0] - x) * -1, p0[1])
        c.line(p1[0], p1[1], x + 1.5 * mm * np.sign(p1[0] - x) * -1, p1[1])
        c.line(x, p0[1] - 1.5 * mm, x, p1[1] + 1.5 * mm)
        for y in (p0[1], p1[1]):
            c.setLineWidth(0.5); c.line(x - 1 * mm, y - 1 * mm, x + 1 * mm, y + 1 * mm); c.setLineWidth(0.25)
        if p1[1] - p0[1] > 7 * mm:
            c.saveState(); c.translate(x - 1 * mm, (p0[1] + p1[1]) / 2); c.rotate(90)
            c.setFont(F, 6.5); c.drawCentredString(0, 0, txt); c.restoreState()


def _plan_body(c, P, T, k, mode='plan', cuts=None, tags=True, only=None):
    """Muros, vanos, capas de la versión acabada y rótulos de espacio (sin ejes ni cotas)."""
    if mode == 'furniture' and P.fin:
        _floor_tints(c, P, T)
    # muros (poché)
    c.setFillGray(0.2); c.setStrokeGray(0); c.setLineWidth(0.5)
    for o, a, b, cc, d in P.walls:
        p, q = T(a, b), T(cc, d)
        c.rect(p[0], p[1], q[0] - p[0], q[1] - p[1], stroke=0, fill=1)
    c.setFillGray(0)
    # vanos
    for op in P.ops:
        p, q = T(op['X0'], op['Y0']), T(op['X1'], op['Y1'])
        c.setFillGray(1); c.rect(p[0], p[1], q[0] - p[0], q[1] - p[1], stroke=0, fill=1); c.setFillGray(0)
        c.setLineWidth(0.18)
        if op['kind'] == 'window':
            if op['orient'] == 'h':
                for f in (0, 0.5, 1):
                    yy = p[1] + f * (q[1] - p[1]); c.line(p[0], yy, q[0], yy)
                c.line(p[0], p[1], p[0], q[1]); c.line(q[0], p[1], q[0], q[1])
            else:
                for f in (0, 0.5, 1):
                    xx = p[0] + f * (q[0] - p[0]); c.line(xx, p[1], xx, q[1])
                c.line(p[0], p[1], q[0], p[1]); c.line(p[0], q[1], q[0], q[1])
        elif op.get('style') == 'opening':  # paso sin puerta: solo las jambas
            c.setLineWidth(0.18)
            if op['orient'] == 'h':
                c.line(p[0], p[1], p[0], q[1]); c.line(q[0], p[1], q[0], q[1])
            else:
                c.line(p[0], p[1], q[0], p[1]); c.line(p[0], q[1], q[0], q[1])
        elif op.get('style') == 'sliding':  # corrediza: dos hojas traslapadas
            c.setLineWidth(0.35)
            if op['orient'] == 'h':
                th = q[1] - p[1]; mid = (p[0] + q[0]) / 2; ov = 0.06 * (q[0] - p[0])
                c.line(p[0], p[1] + 0.3 * th, mid + ov, p[1] + 0.3 * th); c.line(mid - ov, p[1] + 0.7 * th, q[0], p[1] + 0.7 * th)
                c.setLineWidth(0.18); c.line(p[0], p[1], p[0], q[1]); c.line(q[0], p[1], q[0], q[1])
            else:
                th = q[0] - p[0]; mid = (p[1] + q[1]) / 2; ov = 0.06 * (q[1] - p[1])
                c.line(p[0] + 0.3 * th, p[1], p[0] + 0.3 * th, mid + ov); c.line(p[0] + 0.7 * th, mid - ov, p[0] + 0.7 * th, q[1])
                c.setLineWidth(0.18); c.line(p[0], p[1], q[0], p[1]); c.line(p[0], q[1], q[0], q[1])
        elif 'side' in op:  # v5: hoja hacia el lado y con la bisagra que se leyeron de la imagen
            up = op['side'] < 0                     # en la imagen «arriba/izquierda» = -1
            if op['orient'] == 'h':
                r = q[0] - p[0]; left = op.get('hinge', 0) == 0
                hx = p[0] if left else q[0]; hy = q[1] if up else p[1]
                c.setLineWidth(0.35); c.line(hx, hy, hx, hy + (r if up else -r))
                a0 = (0 if left else 90) if up else (270 if left else 180)
            else:
                r = q[1] - p[1]; top = op.get('hinge', 0) == 0; lf = up
                hy = q[1] if top else p[1]; hx = p[0] if lf else q[0]
                c.setLineWidth(0.35); c.line(hx, hy, hx + (-r if lf else r), hy)
                a0 = (180 if lf else 270) if top else (90 if lf else 0)
            c.setLineWidth(0.13); c.setDash(2, 1); c.arc(hx - r, hy - r, hx + r, hy + r, a0, 90); c.setDash()
            if op.get('entry'):
                c.setFont(FB, 5.5)
                ex = (p[0] + q[0]) / 2; ey = (p[1] + q[1]) / 2
                out = op.get('out', 1)
                if op['orient'] == 'h':
                    c.drawCentredString(ex, ey + (-6.5 * mm if out > 0 else 5 * mm), 'ENTRY')
                else:
                    c.drawString(ex + 2.5 * mm, ey - 0.8 * mm, 'ENTRY') if out > 0 else c.drawRightString(ex - 2.5 * mm, ey - 0.8 * mm, 'ENTRY')
        else:  # hoja + arco a 90°
            if op['orient'] == 'h':
                r = q[0] - p[0]; hx, hy = p[0], q[1]
                c.setLineWidth(0.35); c.line(hx, hy, hx, hy + r)
                c.setLineWidth(0.13); c.setDash(2, 1); c.arc(hx - r, hy - r, hx + r, hy + r, 0, 90); c.setDash()
            else:
                r = q[1] - p[1]; hx, hy = q[0], p[1]
                c.setLineWidth(0.35); c.line(hx, hy, hx + r, hy)
                c.setLineWidth(0.13); c.setDash(2, 1); c.arc(hx - r, hy - r, hx + r, hy + r, 0, 90); c.setDash()
        c.setFont(F, 5)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        c.drawCentredString(mx, my + (3 * mm if op['orient'] == 'h' else 0) - 1 * mm, op['tag'])
    # capas de la versión acabada
    if P.fin:
        if mode == 'finishes':
            _floors(c, P, T, k)
        _objects(c, P, T, k, light=(mode not in ('plan', 'furniture')), furniture=False,
                 style='color' if mode == 'furniture' else 'line')
        if mode == 'electrical':
            _electrical(c, P, T, k)
    if cuts and mode == 'plan':
        from . import sections
        sections.draw_cut_marks(c, P, T, cuts, F, FB)
    # espacios
    if not tags:
        return
    def _tag(px, py, nm, sub):
        c.setFillGray(1); c.rect(px - len(nm) * 1.3 * mm / 1.6 - 1 * mm, py - (4 * mm if sub else 1 * mm), len(nm) * 2.6 * mm / 1.6 + 2 * mm,
                                  7.5 * mm if sub else 4.5 * mm, stroke=0, fill=1); c.setFillGray(0)
        c.setFont(FB, 6.5); c.drawCentredString(px, py + 1 * mm, nm)
        if sub:
            c.setFont(F, 6); c.drawCentredString(px, py - 2.5 * mm, sub)
    for r in P.rooms:
        if only is not None and r['id'] not in only:
            continue
        nm = r.get('name', f"S-{r['id']:02d}")
        sub = f"{r['area']:.2f} m²"
        if mode == 'finishes' and r.get('floor'):
            sub = FLOOR_CODE.get(r['floor'], '') + '  ·  ' + sub
        zones = r.get('zones') or []
        if zones:                       # espacio abierto: un rótulo por zona y el área total en la zona principal
            main = max(zones, key=lambda z: 'LIVING' in z[0])
            for z in zones:
                px, py = T(z[1], z[2])
                _tag(px, py, z[0], (sub + ' (open plan)') if z is main else None)
        else:
            px, py = T(r['X'], r['Y'])
            _tag(px, py, nm, sub)


def sheet_plan(c, P, meta, mode='plan', title='FLOOR PLAN', number='A-101', cuts=None):
    x0, y0, x1, y1 = FRAME
    lay = meta.get('plan_layout') or plan_layout(P)
    scale, k, ox, oy = lay['scale'], lay['k'], lay['ox'], lay['oy']
    pan, comp = PANELS.get(mode), COMP.get(mode)
    if lay.get('comp') is None and pan and comp:
        def both(c_, x_, y_, w_, h_):
            y2 = pan(c_, x_, y_, w_, h_) or y_
            rest = h_ - (y_ - y2) - 7 * mm
            if rest > 28 * mm:
                comp(c_, P, x_, y2 - 7 * mm, w_, rest)
        meta['panel'] = both
    else:
        meta['panel'] = pan
    meta['keyplan'] = None
    tb = _frame(c, title, number, meta, f'1:{scale}')

    def T(x, y):
        return ox + x * k, oy + y * k

    # ejes
    c.setDash([6, 2, 1, 2], 0); c.setLineWidth(0.18); c.setStrokeGray(0.45)
    labels_x = [str(i + 1) for i in range(len(P.ax_x))]
    labels_y = [chr(65 + i) for i in range(len(P.ax_y))]
    for x, lb in zip(P.ax_x, labels_x):
        a = T(x, -0.6); b = T(x, P.H + 1.6)
        c.line(a[0], a[1], b[0], b[1])
    for y, lb in zip(P.ax_y, labels_y):
        a = T(-1.6, y); b = T(P.W + 0.6, y)
        c.line(a[0], a[1], b[0], b[1])
    c.setDash(); c.setStrokeGray(0)
    for x, lb in zip(P.ax_x, labels_x):
        px, py = T(x, P.H + 1.6)
        c.circle(px, py + 3.5 * mm, 3.5 * mm); c.setFont(F, 8); c.drawCentredString(px, py + 2.6 * mm, lb)
    for y, lb in zip(P.ax_y, labels_y):
        px, py = T(-1.6, y)
        c.circle(px - 3.5 * mm, py, 3.5 * mm); c.setFont(F, 8); c.drawCentredString(px - 3.5 * mm, py - 1 * mm, lb)

    _plan_body(c, P, T, k, mode, cuts)
    # cotas: cadena entre ejes y total, arriba y a la izquierda
    xs = sorted(set([0.0] + P.ax_x + [P.W]))
    ys = sorted(set([0.0] + P.ax_y + [P.H]))
    top1, top2 = T(0, P.H + 0.7)[1], T(0, P.H + 1.15)[1]
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a > 0.05:
            _dim(c, T(a, P.H), T(b, P.H), top1, f'{b - a:.2f}', True, k)
    _dim(c, T(0, P.H), T(P.W, P.H), top2, f'{P.W:.2f}', True, k)
    l1, l2 = T(-0.7, 0)[0], T(-1.15, 0)[0]
    for a, b in zip(ys[:-1], ys[1:]):
        if b - a > 0.05:
            _dim(c, T(0, a), T(0, b), l1, f'{b - a:.2f}', False, k)
    _dim(c, T(0, 0), T(0, P.H), l2, f'{P.H:.2f}', False, k)
    # título de vista, escala gráfica y panel de datos
    sub = {'plan': 'Finished state: fixed casework and fixtures. Loose furniture on A-105.',
           'furniture': 'Furniture read from the marketing image, redrawn at catalogue sizes. Indicative.',
           'finishes': 'Finishes read from the image texture; confirm against the developer specification.',
           'electrical': 'Schematic proposal by rules, not in the source plan. Verify with a licensed electrician.'}[mode]
    L.view_title(c, x0 + 7 * mm, y0 + 12 * mm, 1, title, f'SCALE 1:{scale}   ·   dimensions in metres', sub)
    L.scale_bar(c, tb - 7 * mm - 5 * k - 6 * mm, y0 + 12 * mm, k)
    if lay.get('comp') and COMP.get(mode):
        cx_, cy_, cw_, ch_ = lay['comp']
        COMP[mode](c, P, cx_, cy_ + ch_, cw_, ch_)
    c.showPage()
    return scale


FLOOR_CODE = {'wood': 'F1', 'tile': 'F2', 'porcelain': 'F3', 'exterior': 'F4'}
FLOOR_LABEL = {'wood': 'F1  Wood-look laminate / plank', 'tile': 'F2  Ceramic tile 60 x 60',
               'porcelain': 'F3  Porcelain / smooth finish', 'exterior': 'F4  Exterior — lawn / paving'}


def _box(c, p, q, fill=None, lw=0.35, dash=None, gray=0):
    c.saveState(); c.setLineWidth(lw); c.setStrokeGray(gray)
    if dash:
        c.setDash(*dash)
    if fill is not None:
        c.setFillGray(fill)
    c.rect(min(p[0], q[0]), min(p[1], q[1]), abs(q[0] - p[0]), abs(q[1] - p[1]), stroke=1, fill=0 if fill is None else 1)
    c.restoreState()


def _objects(c, P, T, k, light=False, furniture=True, style='line'):
    """Mesones/closets, aparatos y (en la lámina de mobiliario) muebles, con bloques paramétricos."""
    from . import blocks
    from .furnish import FIXTURES
    g = 0.55 if light else 0
    have_items = bool(P.items)
    for o in P.objs:
        if have_items and o['kind'] == 'fixture':
            continue                      # los aparatos se dibujan como piezas (bloques) más abajo
        if o['kind'] == 'casework' and style == 'color':
            p, q = T(o['X0'], o['Y0']), T(o['X1'], o['Y1'])
            cw = o.get('casework')
            c.saveState()
            c.setStrokeColorRGB(0.28, 0.27, 0.26); c.setLineWidth(0.4)
            c.setFillColorRGB(*((0.86, 0.76, 0.63) if cw == 'closet' else (0.93, 0.92, 0.90)))
            c.rect(p[0], p[1], q[0] - p[0], q[1] - p[1], stroke=1, fill=1)
            if cw == 'closet':
                w, h = q[0] - p[0], q[1] - p[1]
                c.setLineWidth(0.18); c.setDash(2, 1)
                if w >= h:
                    c.line(p[0] + 1, (p[1] + q[1]) / 2, q[0] - 1, (p[1] + q[1]) / 2)
                else:
                    c.line((p[0] + q[0]) / 2, p[1] + 1, (p[0] + q[0]) / 2, q[1] - 1)
            c.restoreState()
            continue
        p, q = T(o['X0'], o['Y0']), T(o['X1'], o['Y1'])
        w, h = q[0] - p[0], q[1] - p[1]
        cx, cy = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        if o['kind'] == 'furniture':
            if furniture:
                _box(c, p, q, lw=0.15, dash=([1.2, 1.2],), gray=0.55)
            continue
        if o['kind'] == 'casework':
            _box(c, p, q, fill=None if light else 0.93, lw=0.35, gray=g)
            c.saveState(); c.setStrokeGray(g); c.setLineWidth(0.15)
            if o.get('casework') == 'closet':  # barra de colgar
                c.setDash(2, 1)
                if w >= h:
                    c.line(p[0] + 1, cy, q[0] - 1, cy)
                else:
                    c.line(cx, p[1] + 1, cx, q[1] - 1)
            elif o.get('casework') == 'counter':  # gabinete alto (línea a trazos)
                c.setDash(3, 1.5)
                inset = 0.3 * k
                if w >= h:
                    c.line(p[0], cy, q[0], cy)
                else:
                    c.line(cx, p[1], cx, q[1])
            c.restoreState()
            continue
        # aparatos
        f = o.get('fixture', 'fixture')
        c.saveState(); c.setStrokeGray(g); c.setLineWidth(0.3); c.setFillGray(1)
        if f == 'toilet':
            horiz = w >= h
            if horiz:
                tw = 0.22 * k
                c.rect(p[0], p[1] + 0.1 * h, tw, 0.8 * h, fill=1)
                c.ellipse(p[0] + tw, p[1] + 0.12 * h, q[0], q[1] - 0.12 * h, fill=1)
            else:
                th = 0.22 * k
                c.rect(p[0] + 0.1 * w, q[1] - th, 0.8 * w, th, fill=1)
                c.ellipse(p[0] + 0.12 * w, p[1], q[0] - 0.12 * w, q[1] - th, fill=1)
        elif f == 'lavatory':
            c.rect(p[0], p[1], w, h, fill=1)
            c.ellipse(p[0] + 0.18 * w, p[1] + 0.18 * h, q[0] - 0.18 * w, q[1] - 0.18 * h)
        elif f == 'shower':
            c.rect(p[0], p[1], w, h, fill=1)
            c.setLineWidth(0.13); c.line(p[0], p[1], q[0], q[1]); c.line(p[0], q[1], q[0], p[1])
            c.circle(cx, cy, 0.04 * k, fill=1)
        elif f == 'bathtub':
            c.rect(p[0], p[1], w, h, fill=1)
            c.roundRect(p[0] + 0.08 * k, p[1] + 0.08 * k, w - 0.16 * k, h - 0.16 * k, 0.15 * k)
        elif f == 'cooktop':
            c.rect(p[0], p[1], w, h, fill=1)
            for dx in (0.28, 0.72):
                for dy in (0.28, 0.72):
                    c.circle(p[0] + dx * w, p[1] + dy * h, min(w, h) * 0.16)
        elif f == 'sink':
            c.rect(p[0], p[1], w, h, fill=1); c.roundRect(p[0] + 0.15 * w, p[1] + 0.15 * h, 0.7 * w, 0.7 * h, 1.5)
        else:  # fridge / washer
            c.rect(p[0], p[1], w, h, fill=1)
            c.setFillGray(g); c.setFont(F, 4.5)
            c.drawCentredString(cx, cy - 1.5, {'fridge': 'REF', 'washer': 'W'}.get(f, 'EQ'))
            if f == 'washer':
                c.circle(cx, cy, min(w, h) * 0.32)
        c.restoreState()
    for it in P.items:
        if it['type'] not in FIXTURES and style != 'color':
            continue                      # muebles sueltos solo en la lámina de mobiliario
        X, Y = T(it['X'], it['Y'])
        blocks.draw(c, it, X, Y, k, style=style, gray=g, shadow=0.05 if style == 'color' else None)


FLOOR_TINT = {'wood': (0.94, 0.89, 0.81), 'tile': (0.91, 0.92, 0.93), 'porcelain': (0.96, 0.95, 0.93),
              'exterior': (0.88, 0.93, 0.86)}


def _floor_tints(c, P, T):
    for r in P.rooms:
        poly = P.contours.get(r['id'])
        if not poly:
            continue
        col = FLOOR_TINT.get(r.get('floor'), (0.96, 0.96, 0.95))
        c.saveState()
        path = c.beginPath()
        x, y = T(*poly[0]); path.moveTo(x, y)
        for pt in poly[1:]:
            x, y = T(*pt); path.lineTo(x, y)
        path.close()
        c.setFillColorRGB(*col)
        c.drawPath(path, stroke=0, fill=1)
        c.restoreState()


def _floors(c, P, T, k):
    for r in P.rooms:
        poly = P.contours.get(r['id'])
        if not poly or not r.get('floor'):
            continue
        c.saveState()
        path = c.beginPath()
        x, y = T(*poly[0]); path.moveTo(x, y)
        for pt in poly[1:]:
            x, y = T(*pt); path.lineTo(x, y)
        path.close()
        c.clipPath(path, stroke=0, fill=0)
        xs = [T(*pt)[0] for pt in poly]; ys = [T(*pt)[1] for pt in poly]
        a, b, cc, d = min(xs), min(ys), max(xs), max(ys)
        c.setLineWidth(0.12); c.setStrokeGray(0.55)
        fl = r['floor']
        if fl == 'wood':
            stp = 0.18 * k
            if r.get('floor_dir', 'h') == 'h':
                yy = b
                while yy < d:
                    c.line(a, yy, cc, yy); yy += stp
            else:
                xx = a
                while xx < cc:
                    c.line(xx, b, xx, d); xx += stp
        elif fl == 'tile':
            stp = 0.6 * k
            xx = a
            while xx < cc:
                c.line(xx, b, xx, d); xx += stp
            yy = b
            while yy < d:
                c.line(a, yy, cc, yy); yy += stp
        elif fl == 'exterior':  # césped / exterior: trazos cortos alternados
            stp = 0.35 * k
            yy, row = b, 0
            while yy < d:
                xx = a + (stp / 2 if row % 2 else 0)
                while xx < cc:
                    c.line(xx, yy, xx + 0.07 * k, yy + 0.1 * k); c.line(xx + 0.07 * k, yy, xx + 0.1 * k, yy + 0.08 * k); xx += stp
                yy += stp; row += 1
        else:  # porcelana: punteado ligero
            stp = 0.4 * k
            yy = b
            while yy < d:
                xx = a
                while xx < cc:
                    c.circle(xx, yy, 0.25, stroke=1, fill=0); xx += stp
                yy += stp
        c.restoreState()


def _sym(c, t, x, y):
    c.saveState(); c.setLineWidth(0.35); c.setFillGray(1)
    if t == 'light':
        c.circle(x, y, 2.0 * mm, fill=1); c.line(x - 1.4 * mm, y - 1.4 * mm, x + 1.4 * mm, y + 1.4 * mm)
        c.line(x - 1.4 * mm, y + 1.4 * mm, x + 1.4 * mm, y - 1.4 * mm)
    elif t in ('outlet', 'gfci'):
        c.circle(x, y, 1.3 * mm, fill=1); c.line(x - 0.5 * mm, y - 0.8 * mm, x - 0.5 * mm, y + 0.8 * mm)
        c.line(x + 0.5 * mm, y - 0.8 * mm, x + 0.5 * mm, y + 0.8 * mm)
        if t == 'gfci':
            c.setFillGray(0); c.setFont(FB, 3.5); c.drawString(x + 1.5 * mm, y - 0.6 * mm, 'GFCI')
    elif t == 'switch':
        c.setFillGray(0); c.setFont(FB, 6); c.drawCentredString(x, y - 1 * mm, 'S')
    c.restoreState()


def _electrical(c, P, T, k):
    for p in P.elec:
        x, y = T(p['X'], p['Y'])
        _sym(c, p['t'], x, y)


def _legend(c, mode, tb, P=None):
    x0, y0, x1, y1 = FRAME
    lx, ly = tb + 5 * mm, y1 - 30 * mm
    c.setFont(FB, 8.5); c.drawString(lx, ly, 'LEGEND' if mode != 'furniture' else 'FURNITURE & FIXTURES')
    c.setFont(F, 7)
    if mode == 'furniture':
        from collections import Counter
        from .furnish import NAMES, FURNITURE, FIXTURES
        cnt = Counter(it['type'] for it in (P.items if P else []))
        rows = [t for t in FURNITURE + FIXTURES if cnt.get(t)]
        for i, t in enumerate(rows[:16]):
            c.drawString(lx, ly - (6 + 4.4 * i) * mm, NAMES.get(t, t))
            c.drawRightString(lx + 52 * mm, ly - (6 + 4.4 * i) * mm, str(cnt[t]))
        return
    if mode == 'finishes':
        items = list(FLOOR_LABEL.values())
        for i, it in enumerate(items):
            c.drawString(lx, ly - (6 + 5 * i) * mm, it)
    else:
        for i, (t, txt) in enumerate((('light', 'Ceiling light point'), ('outlet', 'Duplex outlet (perimeter, ~3 m)'),
                                      ('gfci', 'GFCI outlet (counter ~1.2 m / lavatory)'), ('switch', 'Switch at door'))):
            yy = ly - (7 + 6 * i) * mm
            _sym(c, t, lx + 2 * mm, yy + 1 * mm)
            c.setFillGray(0); c.setFont(F, 7); c.drawString(lx + 8 * mm, yy, txt)


# ------------------------------------------------------------------------------------------ paneles
def _swatch(c, x, y, w, h, fl):
    """Muestra de acabado de piso (la misma trama de la lámina A-103)."""
    c.saveState()
    c.setStrokeGray(0.25); c.setLineWidth(0.4); c.setFillColorRGB(*FLOOR_TINT.get(fl, (0.96, 0.96, 0.95)))
    c.rect(x, y, w, h, stroke=1, fill=1)
    p = c.beginPath(); p.rect(x, y, w, h); c.clipPath(p, stroke=0, fill=0)
    c.setLineWidth(0.15); c.setStrokeGray(0.45)
    if fl == 'wood':
        yy = y
        while yy < y + h:
            c.line(x, yy, x + w, yy); yy += 1.1 * mm
    elif fl == 'tile':
        for i in range(1, 4):
            c.line(x + i * w / 4, y, x + i * w / 4, y + h)
        for j in range(1, 3):
            c.line(x, y + j * h / 3, x + w, y + j * h / 3)
    elif fl == 'exterior':
        for i in range(6):
            for j in range(3):
                xx, yy = x + (i + 0.3 + 0.5 * (j % 2)) * w / 6, y + (j + 0.4) * h / 3
                c.line(xx, yy, xx + 0.5 * mm, yy + 0.8 * mm)
    else:
        for i in range(6):
            for j in range(3):
                c.circle(x + (i + 0.5) * w / 6, y + (j + 0.5) * h / 3, 0.18, stroke=1, fill=0)
    c.restoreState()


def _panel_plan(c, x, y, w, h):
    """Leyenda de la planta: cada símbolo dibujado como en la lámina."""
    from . import blocks
    y = L.header(c, x, y, w, 'Legend')
    rh = 6.6 * mm
    sx, sw, tx = x, 15 * mm, x + 19 * mm

    def row(txt, fn):
        nonlocal y
        if y - rh < (y - h):
            return
        c.saveState(); fn(sx, y - 4.4 * mm); c.restoreState()
        c.setFont(F, 6.4); c.setFillGray(0.1); c.drawString(tx, y - 3.6 * mm, txt)
        y -= rh

    def wall(px, py):
        c.setFillGray(0.2); c.rect(px, py, sw, 2.2 * mm, stroke=0, fill=1)

    def door(px, py):
        c.setFillGray(0.2); c.rect(px, py, 2.5 * mm, 1.6 * mm, stroke=0, fill=1); c.rect(px + sw - 2.5 * mm, py, 2.5 * mm, 1.6 * mm, stroke=0, fill=1)
        r = sw - 5 * mm
        c.setLineWidth(0.35); c.line(px + 2.5 * mm, py + 1.6 * mm, px + 2.5 * mm, py + 1.6 * mm + r * 0.45)
        c.setLineWidth(0.13); c.setDash(2, 1); c.arc(px + 2.5 * mm - r, py + 1.6 * mm - r, px + 2.5 * mm + r, py + 1.6 * mm + r, 0, 25); c.setDash()

    def sliding(px, py):
        c.setFillGray(0.2); c.rect(px, py, 2.5 * mm, 1.6 * mm, stroke=0, fill=1); c.rect(px + sw - 2.5 * mm, py, 2.5 * mm, 1.6 * mm, stroke=0, fill=1)
        c.setLineWidth(0.4); c.line(px + 2.5 * mm, py + 0.5 * mm, px + sw * 0.6, py + 0.5 * mm); c.line(px + sw * 0.4, py + 1.1 * mm, px + sw - 2.5 * mm, py + 1.1 * mm)

    def window(px, py):
        c.setFillGray(0.2); c.rect(px, py, 2.5 * mm, 1.8 * mm, stroke=0, fill=1); c.rect(px + sw - 2.5 * mm, py, 2.5 * mm, 1.8 * mm, stroke=0, fill=1)
        c.setLineWidth(0.18)
        for f in (0, 0.5, 1):
            c.line(px + 2.5 * mm, py + f * 1.8 * mm, px + sw - 2.5 * mm, py + f * 1.8 * mm)

    def axis(px, py):
        c.setLineWidth(0.18); c.setStrokeGray(0.45); c.setDash([6, 2, 1, 2], 0); c.line(px + 4 * mm, py + 1 * mm, px + sw, py + 1 * mm); c.setDash()
        c.setStrokeGray(0); c.circle(px + 2 * mm, py + 1 * mm, 2 * mm); c.setFont(F, 5.5); c.drawCentredString(px + 2 * mm, py - 0.2 * mm, '1')

    def dim(px, py):
        c.setLineWidth(0.25); c.line(px, py + 0.6 * mm, px + sw, py + 0.6 * mm)
        for xx in (px + 1 * mm, px + sw - 1 * mm):
            c.setLineWidth(0.5); c.line(xx - 0.8 * mm, py - 0.2 * mm, xx + 0.8 * mm, py + 1.4 * mm)
        c.setFont(F, 5); c.drawCentredString(px + sw / 2, py + 1.2 * mm, '2.40')

    def tag(px, py):
        c.setFont(FB, 5.4); c.drawCentredString(px + sw / 2, py + 1.4 * mm, 'BEDROOM')
        c.setFont(F, 5); c.drawCentredString(px + sw / 2, py - 0.6 * mm, '10.6 m²')

    def cut(px, py):
        c.setLineWidth(0.5); c.setDash([5, 1.5, 1, 1.5], 0); c.line(px + 3 * mm, py + 1 * mm, px + sw, py + 1 * mm); c.setDash()
        c.circle(px + 1.8 * mm, py + 1 * mm, 1.8 * mm); c.setFont(FB, 5.5); c.drawCentredString(px + 1.8 * mm, py - 0.1 * mm, 'A')
        c.line(px + sw - 1 * mm, py + 1 * mm, px + sw - 1 * mm, py + 3.2 * mm)

    def fixture(px, py):
        blocks.draw(c, dict(type='toilet', w=0.4, d=0.7, ang=90.0), px + 4 * mm, py + 1 * mm, 9 * mm)
        blocks.draw(c, dict(type='lavatory', w=0.5, d=0.45, ang=0.0), px + 11.5 * mm, py + 1 * mm, 9 * mm)

    row('Wall, cut', wall)
    row('Door with swing', door)
    row('Sliding door', sliding)
    row('Window', window)
    row('Axis with reference bubble', axis)
    row('Dimension (m, to wall axes)', dim)
    row('Space name and net area', tag)
    row('Section cut, view direction', cut)
    row('Fixture / casework (block)', fixture)
    y -= 2 * mm
    y = L.wrap(c, 'Openings tagged D-xx (doors) and W-xx (windows) as in the schedule on A-102.', x, y, w, size=6.0)
    return y

def _panel_finishes(c, x, y, w, h):
    y = L.header(c, x, y, w, 'Floor finishes')
    for code, txt in FLOOR_LABEL.items():
        _swatch(c, x, y - 5.4 * mm, 11 * mm, 6 * mm, code)
        c.setFont(FB, 6.6); c.setFillGray(0.1); c.drawString(x + 14 * mm, y - 2.6 * mm, txt.split('  ')[0])
        c.setFont(F, 6.2); c.drawString(x + 14 * mm, y - 5.4 * mm, txt.split('  ', 1)[1])
        y -= 9 * mm
    y -= 2 * mm
    y = L.wrap(c, 'Read from the texture of the image: wood grain, tile joints or a plain surface. '
                  'Confirm codes, formats and skirting against the developer specification.', x, y, w, size=6.0)
    return y

def _panel_electrical(c, x, y, w, h):
    y = L.header(c, x, y, w, 'Legend')
    for t, txt in (('light', 'Ceiling light point'), ('outlet', 'Duplex outlet, perimeter ~3 m'),
                   ('gfci', 'GFCI outlet (counter ~1.2 m / lavatory)'), ('switch', 'Switch at door')):
        _sym(c, t, x + 3 * mm, y - 2.4 * mm)
        c.setFillGray(0.1); c.setFont(F, 6.6); c.drawString(x + 10 * mm, y - 3.4 * mm, txt)
        y -= 7 * mm
    y -= 2 * mm
    y = L.wrap(c, 'A rule-based proposal for quoting and discussion: points per room, outlets every ~3 m of wall, '
                  'GFCI near water, a switch at each door. It is NOT taken from the source plan; '
                  'circuits, loads and code compliance must be designed by a licensed electrician.', x, y, w, size=6.0)
    return y

def _panel_furniture(c, x, y, w, h):
    from . import blocks
    y = L.header(c, x, y, w, 'Reading this sheet')
    for st, txt in (('solid', 'Solid: piece read from the marketing image'),
                    ('dash', 'Dashed: assumed (every bathroom gets a toilet and a lavatory) or unidentified')):
        c.saveState(); c.setLineWidth(0.5); c.setStrokeGray(0.2); c.setFillColorRGB(0.80, 0.83, 0.86)
        if st == 'dash':
            c.setDash(1.4, 1); c.setFillGray(1)
        c.roundRect(x, y - 5 * mm, 11 * mm, 5 * mm, 1.2, stroke=1, fill=1); c.restoreState()
        L.wrap(c, txt, x + 14 * mm, y - 1.8 * mm, w - 14 * mm, size=6.2)
        y -= 9 * mm
    y -= 1 * mm
    y = L.wrap(c, 'Each piece is redrawn as a standard block at a catalogue size (beds 0.90–2.00 m wide, chairs 0.46 m, '
                  'toilets 0.40 × 0.70 m) and turned to face the way it faces in the image: headboards and toilet '
                  'tanks against the wall, chairs towards the table. Furniture is indicative, not part of the delivery.',
               x, y, w, size=6.0)
    return y

def _panel_schedules(c, x, y, w, h):
    P = _CUR.get('P')
    y = L.header(c, x, y, w, 'Area summary')
    if not P:
        return
    inter = [r for r in P.rooms if not r.get('exterior')]
    ext = [r for r in P.rooms if r.get('exterior')]
    net = sum(r['area'] for r in inter)
    big = [('Net interior', f'{net:.2f} m²'), ('Exterior (patio / balcony)', f"{sum(r['area'] for r in ext):.2f} m²" if ext else '—'),
           ('Footprint', f'{P.W:.2f} × {P.H:.2f} m'), ('Spaces', str(len(inter)))]
    for lb, v in big:
        L.label(c, x, y - 2 * mm, lb)
        c.setFont(FB, 12); c.setFillGray(0.05); c.drawString(x, y - 7.6 * mm, v)
        y -= 11.5 * mm
    meta = _CUR.get('meta') or {}
    y -= 1 * mm
    y = L.header(c, x, y, w, 'Scale evidence')
    y = L.wrap(c, (meta.get('assume1') or ('1. ' + meta.get('scaled_from', ''))).split('. ', 1)[-1], x, y, w, size=6.1, leading=7.4)
    return y

def _panel_cover(c, x, y, w, h):
    y = L.header(c, x, y, w, 'About this set')
    y = L.wrap(c, 'Generated automatically from one marketing floor-plan image by Zero Mouse Clicks: three small '
              'segmentation models read walls, openings, casework and furniture; explicit drafting rules square the '
              'geometry, set the scale and draw every sheet. Nothing was measured on site.\n'
              'Open source (Apache-2.0). Every rule can be read and changed.', x, y, w, size=6.3, leading=8.2)
    meta = _CUR.get('meta') or {}
    y -= 3 * mm
    y = L.header(c, x, y, w, 'Scale evidence')
    y = L.wrap(c, (meta.get('assume1') or ('1. ' + meta.get('scaled_from', ''))).split('. ', 1)[-1], x, y, w, size=6.1, leading=7.4)
    return y


PANELS = {'plan': _panel_plan, 'finishes': _panel_finishes, 'electrical': _panel_electrical, 'furniture': _panel_furniture}
_CUR = {}


# ------------------------------------------------------------------------------------------ paneles de datos (junto a la planta)

def _short(t, n):
    t = str(t)
    return t if len(t) <= n else t[:n - 1] + '…'


def _fit_widths(fracs, w):
    s_ = sum(fracs)
    return [w * f / s_ for f in fracs]


def _comp_plan(c, P, x, y, w, h):
    bottom = y - h
    y = L.header(c, x, y, w, 'Room schedule')
    rows = [(f"S-{r['id']:02d}", _short(r.get('name', '—'), 26), f"{r['area']:.2f}") for r in P.rooms if not r.get('exterior')]
    tot = sum(r['area'] for r in P.rooms if not r.get('exterior'))
    y = L.table(c, x, y, ['Code', 'Space', 'm²'], rows, _fit_widths([0.17, 0.6, 0.23], w), align=['l', 'l', 'r'],
                total=('', 'Net interior', f'{tot:.2f}'), max_h=y - bottom - 21 * mm)
    ext = [r for r in P.rooms if r.get('exterior')]
    if ext and y - bottom > 40 * mm:
        y = L.table(c, x, y - 2 * mm, ['Code', 'Exterior', 'm²'], [(f"S-{r['id']:02d}", r.get('name', '—'), f"{r['area']:.2f}") for r in ext],
                    _fit_widths([0.17, 0.6, 0.23], w), align=['l', 'l', 'r'], max_h=y - bottom - 28 * mm)
    y -= 3 * mm
    if y - bottom < 18 * mm:
        return y
    y = L.header(c, x, y, w, 'Openings')
    d = [o for o in P.ops if o['kind'] == 'door']; wn = [o for o in P.ops if o['kind'] == 'window']
    widths = ', '.join(sorted({f"{o['width']:.2f}" for o in d})[:5])
    for lb, v in (('Doors', f"{len(d)}  ·  widths {widths} m" if d else '—'),
                  ('Windows', f"{len(wn)}  ·  total width {sum(o['width'] for o in wn):.2f} m" if wn else '—')):
        L.label(c, x, y, lb); c.setFont(F, 6.4); c.setFillGray(0.1); c.drawString(x + 15 * mm, y, v); y -= 4.6 * mm
    L.wrap(c, 'Full door and window schedule with types on A-102.', x, y - 1 * mm, w, size=6.0, gray=0.4)


def _perimeter(poly):
    return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))) if poly else 0.0


def _comp_finishes(c, P, x, y, w, h):
    bottom = y - h
    y = L.header(c, x, y, w, 'Finish quantities')
    agg = {}
    for r in P.rooms:
        fl = r.get('floor') or '—'
        a = agg.setdefault(fl, dict(n=0, area=0.0, per=0.0))
        a['n'] += 1; a['area'] += r['area']; a['per'] += _perimeter(P.contours.get(r['id']))
    rows = [(FLOOR_CODE.get(fl, '—'), _short(FLOOR_LABEL.get(fl, fl).split('  ', 1)[-1], 20), v['n'], f"{v['area']:.2f}", f"{v['per']:.1f}")
            for fl, v in sorted(agg.items(), key=lambda kv: FLOOR_CODE.get(kv[0], 'Z'))]
    y = L.table(c, x, y, ['Code', 'Finish', 'Sp.', 'm²', 'Perim. m'], rows, _fit_widths([0.13, 0.42, 0.1, 0.17, 0.18], w),
                align=['l', 'l', 'r', 'r', 'r'], total=('', 'Total', sum(v['n'] for v in agg.values()),
                                                         f"{sum(v['area'] for v in agg.values()):.2f}", f"{sum(v['per'] for v in agg.values()):.1f}"))
    y -= 3 * mm
    if y - bottom < 30 * mm:
        return y
    y = L.header(c, x, y, w, 'By space')
    rows = [(f"S-{r['id']:02d}", _short(r.get('name', '—'), 22), FLOOR_CODE.get(r.get('floor', ''), '—'), f"{r['area']:.2f}") for r in P.rooms]
    y = L.table(c, x, y, ['Code', 'Space', 'Floor', 'm²'], rows, _fit_widths([0.17, 0.5, 0.15, 0.18], w), align=['l', 'l', 'l', 'r'],
                max_h=y - bottom - 8 * mm)
    L.wrap(c, 'Perimeter = room outline, before deducting door widths; use it to estimate skirting.', x, y - 1 * mm, w, size=5.8, gray=0.4)


def _comp_electrical(c, P, x, y, w, h):
    y = L.header(c, x, y, w, 'Electrical quantities')
    names = {r['id']: r.get('name', '—') for r in P.rooms}
    cnt = {}
    for p_ in P.elec:
        cnt.setdefault(p_.get('room'), dict(light=0, outlet=0, gfci=0, switch=0))[p_['t']] += 1
    rows = [(_short(names.get(rid, '—'), 20), v['light'], v['outlet'], v['gfci'], v['switch']) for rid, v in cnt.items()]
    tot = [sum(v[t] for v in cnt.values()) for t in ('light', 'outlet', 'gfci', 'switch')]
    L.table(c, x, y, ['Space', 'Light', 'Outl.', 'GFCI', 'Sw.'], rows, _fit_widths([0.44, 0.14, 0.14, 0.14, 0.14], w),
            align=['l', 'r', 'r', 'r', 'r'], total=['Total'] + tot, max_h=h - 6 * mm, more='')


def _comp_furniture(c, P, x, y, w, h):
    from . import blocks
    from .furnish import NAMES, FURNITURE, FIXTURES
    y = L.header(c, x, y, w, 'Furniture & fixtures')
    groups = {}
    for it in P.items:
        g = groups.setdefault(it['type'], [])
        g.append(it)
    order = [t for t in FURNITURE + FIXTURES if t in groups]
    rh = 8.6 * mm
    bottom = y - h + 3 * mm
    c.setFont(FB, 5.6); c.setFillGray(MUTE_G)
    c.drawString(x + 13 * mm, y, 'ITEM'); c.drawRightString(x + w - 22 * mm, y, 'QTY'); c.drawRightString(x + w, y, 'TYPICAL SIZE')
    y -= 2 * mm
    for t in order:
        if y - rh < bottom:
            break
        its = groups[t]
        sz = max(its, key=lambda i: i['w'] * i['d'])
        icon_k = min(9 * mm / max(sz['w'], sz['d']), 14 * mm)
        blocks.draw(c, dict(sz, ang=0.0, src='model'), x + 5 * mm, y - rh / 2, icon_k, style='color')
        c.setFillGray(0.1); c.setFont(F, 6.6); c.drawString(x + 13 * mm, y - rh / 2 - 2, NAMES.get(t, t))
        c.setFont(FB, 6.8); c.drawRightString(x + w - 22 * mm, y - rh / 2 - 2, str(len(its)))
        c.setFont(F, 6.2); c.drawRightString(x + w, y - rh / 2 - 2, f"{sz['w']:.2f} × {sz['d']:.2f}")
        c.setStrokeGray(0.85); c.setLineWidth(0.25); c.line(x, y - rh, x + w, y - rh)
        y -= rh
    n_as = sum(1 for it in P.items if it.get('src') == 'assumed')
    if n_as:
        L.wrap(c, f'{n_as} piece(s) assumed (dashed).', x, y - 3 * mm, w, size=5.8, gray=0.4)


MUTE_G = 0.42
COMP = {'plan': _comp_plan, 'finishes': _comp_finishes, 'electrical': _comp_electrical, 'furniture': _comp_furniture}


# ------------------------------------------------------------------------------------------ A-102 cuadros
def _types(P):
    """Tipos de puerta y ventana: mismo uso, estilo y ancho (redondeado a 5 cm)."""
    tp = {}
    for o in P.ops:
        kind = 'Sliding door' if o.get('style') == 'sliding' else 'Opening' if o.get('style') == 'opening' else \
            'Entry door' if o.get('entry') else 'Door' if o['kind'] == 'door' else 'Window'
        wdt = round(o['width'] * 20) / 20
        key = (o['kind'], kind, wdt)
        tp.setdefault(key, []).append(o)
    out = []
    nd = nw = 0
    for (k_, kind, wdt), os_ in sorted(tp.items(), key=lambda kv: (kv[0][0] != 'door', kv[0][2])):
        if k_ == 'door':
            nd += 1; code = f'D{nd}'
        else:
            nw += 1; code = f'W{nw}'
        for o in os_:
            o['type_code'] = code
        out.append(dict(code=code, kind=kind, width=wdt, height=2.10 if k_ == 'door' else 1.20, n=len(os_), door=k_ == 'door',
                        style=os_[0].get('style')))
    return out


def _elev(c, t, x, y, k):
    """Alzado esquemático de un tipo (k = puntos por metro); (x, y) = esquina inferior izquierda del vano."""
    w, h = t['width'] * k, t['height'] * k
    c.saveState(); c.setStrokeGray(0.1)
    sill = 0 if t['door'] else 1.0 * k
    c.setLineWidth(0.25); c.setStrokeGray(0.5); c.line(x - 3 * mm, y, x + w + 3 * mm, y)   # piso
    c.setStrokeGray(0.1); c.setLineWidth(0.6); c.rect(x, y + sill, w, h)
    c.setLineWidth(0.25)
    if t['door'] and t['style'] != 'opening':
        f = 0.05 * k
        c.rect(x + f, y + sill, w - 2 * f, h - f)
        if t['style'] == 'sliding':
            c.line(x + w / 2, y + sill, x + w / 2, y + sill + h - f)
        else:
            c.circle(x + w - 2.2 * f, y + sill + 1.0 * k, 0.6, stroke=1, fill=0)
            c.setDash(1.5, 1); c.line(x + f, y + sill + h - f, x + w - f, y + sill + h / 2); c.line(x + f, y + sill, x + w - f, y + sill + h / 2); c.setDash()
    elif not t['door']:
        n = max(1, int(round(t['width'] / 0.9)))
        for i in range(1, n):
            c.line(x + i * w / n, y + sill, x + i * w / n, y + sill + h)
        c.rect(x + 0.04 * k, y + sill + 0.04 * k, w - 0.08 * k, h - 0.08 * k)
    # cotas
    c.setFont(F, 5.2); c.setFillGray(0.2)
    c.line(x, y - 2 * mm, x + w, y - 2 * mm)
    c.drawCentredString(x + w / 2, y - 4.6 * mm, f"{t['width']:.2f}")
    c.line(x - 2 * mm, y + sill, x - 2 * mm, y + sill + h)
    c.saveState(); c.translate(x - 2.8 * mm, y + sill + h / 2); c.rotate(90); c.drawCentredString(0, 0, f"{t['height']:.2f}"); c.restoreState()
    c.restoreState()


def sheet_schedules(c, P, meta, scale):
    x0, y0, x1, y1 = FRAME
    meta['panel'] = _panel_schedules
    meta['keyplan'] = {}
    _CUR['P'] = P
    tb = _frame(c, 'AREAS & OPENINGS SCHEDULES', 'A-102', meta, 'as noted')
    dx0, dy0, dx1, dy1 = L.drawing_area()
    gut = 7 * mm
    W = dx1 - dx0
    c1w, c2w = W * 0.35, W * 0.36
    c3w = W - c1w - c2w - 2 * gut
    cx1, cx2, cx3 = dx0, dx0 + c1w + gut, dx0 + c1w + c2w + 2 * gut
    top = dy1 - 2 * mm

    # columna 1: cuadro de áreas y diagrama de áreas
    inter = [r for r in P.rooms if not r.get('exterior')]
    exter = [r for r in P.rooms if r.get('exterior')]
    tot = sum(r['area'] for r in inter) or 1.0
    y = L.header(c, cx1, top, c1w, 'Area & finish schedule (m²)')
    rows = [(f"S-{r['id']:02d}", _short(r.get('name', '—'), 22), FLOOR_CODE.get(r.get('floor', ''), '—'), f"{r['area']:.2f}",
             (r['area'] / tot, f"{100 * r['area'] / tot:.1f} %")) for r in inter]
    y = L.table(c, cx1, y, ['Code', 'Space', 'Floor', 'Area', 'Share'], rows, _fit_widths([0.13, 0.4, 0.11, 0.14, 0.22], c1w),
                align=['l', 'l', 'l', 'r', 'r'], total=('', 'Net interior', '', f'{tot:.2f}', '100 %'), bars=4,
                max_h=(top - dy0) * 0.62, more='')
    if exter:
        y = L.table(c, cx1, y - 1 * mm, ['Code', 'Exterior', 'Floor', 'Area', ''],
                    [(f"S-{r['id']:02d}", r.get('name', '—'), FLOOR_CODE.get(r.get('floor', ''), '—'), f"{r['area']:.2f}", '') for r in exter],
                    _fit_widths([0.13, 0.4, 0.11, 0.14, 0.22], c1w), align=['l', 'l', 'l', 'r', 'r'])
    y -= 4 * mm
    y = L.header(c, cx1, y, c1w, 'Area diagram')
    pal = [(0.86, 0.91, 0.96), (0.93, 0.89, 0.82), (0.88, 0.93, 0.87), (0.95, 0.88, 0.88), (0.91, 0.88, 0.95), (0.95, 0.93, 0.84)]
    fills = {r['id']: pal[i % len(pal)] for i, r in enumerate(P.rooms)}
    L.key_plan(c, P, cx1, dy0 + 2 * mm, c1w, max(20 * mm, y - dy0 - 6 * mm), codes=True, fills=fills)

    # columna 2: cuadro de puertas y ventanas + tipos
    types = _types(P)
    y = L.header(c, cx2, top, c2w, 'Door & window schedule')
    ops = [(o['tag'], o.get('type_code', ''), ('Sliding door' if o.get('style') == 'sliding' else 'Opening' if o.get('style') == 'opening'
             else 'Entry door' if o.get('entry') else 'Door') if o['kind'] == 'door' else 'Window', f"{o['width']:.2f}",
            '2.10' if o['kind'] == 'door' else '1.20*') for o in P.ops]
    maxrows = int((top - dy0 - 70 * mm) / (4.3 * mm))
    shown = ops[:max(6, maxrows)]
    y = L.table(c, cx2, y, ['Tag', 'Type', 'Kind', 'Width', 'Height'], shown, _fit_widths([0.15, 0.13, 0.36, 0.17, 0.19], c2w),
                align=['l', 'l', 'l', 'r', 'r'], size=6.2, rh=4.1 * mm)
    if len(shown) < len(ops):
        L.wrap(c, f'… and {len(ops) - len(shown)} more (same types).', cx2, y, c2w, size=5.8, gray=0.4); y -= 4 * mm
    L.wrap(c, 'Widths measured on the image; heights assumed (* window head 2.20, sill 1.00).', cx2, y, c2w, size=5.8, gray=0.4)
    y -= 6 * mm
    y = L.header(c, cx2, y, c2w, 'Types (elevations, schematic)')
    avail_h = y - dy0 - 8 * mm
    ke = min(14 * mm / 1.0, avail_h / 2.6)          # escala de los alzados
    xx, yy = cx2 + 6 * mm, y - 3 * mm - 2.35 * ke
    row_h = 2.35 * ke + 13 * mm
    for t in types:
        wpt = t['width'] * ke
        if xx + wpt > cx2 + c2w:
            xx = cx2 + 6 * mm; yy -= row_h
        if yy < dy0 + 8 * mm:
            break
        _elev(c, t, xx, yy, ke)
        c.setFont(FB, 6.4); c.setFillGray(0.05); c.drawString(xx, yy - 8.2 * mm, f"{t['code']}  ×{t['n']}")
        c.setFont(F, 5.6); c.setFillGray(0.35); c.drawString(xx, yy - 10.8 * mm, t['kind'])
        xx += max(wpt, 16 * mm) + 9 * mm

    # columna 3: aparatos, mobiliario y supuestos
    y = L.header(c, cx3, top, c3w, 'Fixtures & casework')
    from collections import Counter
    cnt = Counter((o.get('fixture') or o.get('casework') or o['kind']) for o in P.objs if o['kind'] != 'furniture' and o['kind'] != 'fixture')
    cnt.update(it['type'] for it in P.items if it['type'] in ('fridge', 'cooktop', 'sink', 'washer', 'toilet', 'lavatory', 'shower', 'bathtub'))
    rows2 = [(k_.replace('_', ' ').upper(), v) for k_, v in sorted(cnt.items())]
    y = L.table(c, cx3, y, ['Item', 'Qty'], rows2, _fit_widths([0.78, 0.22], c3w), align=['l', 'r'], max_h=(top - dy0) * 0.3, more='')
    y -= 3 * mm
    y = L.header(c, cx3, y, c3w, 'Furniture (indicative)')
    from .furnish import NAMES, FURNITURE
    fc = Counter(it['type'] for it in P.items if it['type'] in FURNITURE)
    y = L.table(c, cx3, y, ['Item', 'Qty'], [(NAMES.get(k_, k_), v) for k_, v in sorted(fc.items())] or [('—', '')],
                _fit_widths([0.78, 0.22], c3w), align=['l', 'r'], max_h=(top - dy0) * 0.3, more='on A-105')
    y -= 3 * mm
    y = L.header(c, cx3, y, c3w, 'Assumptions')
    for i, s_ in enumerate([
        meta.get('assume1') or (f"1. Scale derived from {P.m.get('scale_from', 'declared area')}."),
        '2. Walls, doors and windows read from the image by small on-device models and squared by rules.',
        '3. Door height 2.10 m, window head 2.20 m, clear height 2.45 m: assumed.',
        '4. Room names read from the labels printed on the plan when present; otherwise inferred by rules.',
        '5. Electrical points (A-104) and furniture (A-105) are proposals by rules, not data from the source.',
        '6. ' + NOTE + '.']):
        y = L.wrap(c, s_, cx3, y, c3w, size=5.9, leading=7.1) - 1.2
        if y < dy0:
            break
    L.view_title(c, x0 + 7 * mm, y0 + 12 * mm, 2, 'AREAS & OPENINGS SCHEDULES', 'Areas in m², net to the inner face of walls',
                 'Elevations of types are schematic, not to a stated scale.')
    c.showPage()


# ------------------------------------------------------------------------------------------ G-001 portada
def _img_reader(img):
    import io
    import cv2
    from reportlab.lib.utils import ImageReader
    h, w = img.shape[:2]
    f = min(1.0, 1400.0 / max(h, w))
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return ImageReader(io.BytesIO(buf.tobytes())), img.shape[1], img.shape[0]


def sheet_cover(c, P, meta, index, image=None):
    x0, y0, x1, y1 = FRAME
    meta['panel'] = _panel_cover
    meta['keyplan'] = {}
    tb = _frame(c, 'COVER SHEET & INDEX', 'G-001', meta, 'n.t.s.')
    dx0, dy0, dx1, dy1 = L.drawing_area()
    W = dx1 - dx0
    y = dy1 - 6 * mm
    c.setFont(FB, 7.5); c.setFillColorRGB(*ACC)
    c.drawString(dx0, y, 'A R C H I T E C T U R A L   P L A N   S E T')
    c.setFillGray(0.04); c.setFont(FB, 26)
    proj = meta['project']
    while c.stringWidth(proj, FB, 26) > W and len(proj) > 6:
        proj = proj[:-2]
    c.drawString(dx0, y - 12 * mm, proj)
    c.setFont(F, 13); c.setFillGray(0.25); c.drawString(dx0, y - 19.5 * mm, meta['unit'])
    y -= 27 * mm
    c.setStrokeGray(0.05); c.setLineWidth(1.2); c.line(dx0, y, dx1, y)
    # cifras
    inter = [r for r in P.rooms if not r.get('exterior')]
    net = sum(r['area'] for r in inter)
    ext = sum(r['area'] for r in P.rooms if r.get('exterior'))
    nd = sum(1 for o in P.ops if o['kind'] == 'door'); nw = sum(1 for o in P.ops if o['kind'] == 'window')
    tiles = [('Net interior area', f'{net:.1f} m²'), ('Exterior', f'{ext:.1f} m²' if ext else '—'),
             ('Footprint', f'{P.W:.2f} × {P.H:.2f} m'), ('Spaces', str(len(inter))), ('Doors / windows', f'{nd} / {nw}')]
    tw = W / len(tiles)
    for i, (lb, v) in enumerate(tiles):
        tx = dx0 + i * tw
        L.label(c, tx, y - 5 * mm, lb, size=5.8)
        c.setFont(FB, 15); c.setFillGray(0.04); c.drawString(tx, y - 12.5 * mm, v)
        if i:
            c.setStrokeGray(0.75); c.setLineWidth(0.4); c.line(tx - 3 * mm, y - 2 * mm, tx - 3 * mm, y - 14 * mm)
    y -= 18 * mm
    c.setStrokeGray(0.75); c.setLineWidth(0.4); c.line(dx0, y, dx1, y)
    y -= 6 * mm
    # imagen de origen (izquierda) e índice (derecha)
    lw = W * 0.55
    rx = dx0 + lw + 9 * mm
    rw = dx1 - rx
    L.header(c, dx0, y, lw, 'Source: marketing floor plan')
    box_h = y - 8 * mm - (dy0 + 2 * mm)
    if image is not None:
        try:
            rd, iw, ih = _img_reader(image)
            f = min(lw / iw, box_h / ih)
            c.drawImage(rd, dx0 + (lw - iw * f) / 2, y - 7 * mm - ih * f, iw * f, ih * f)
            c.setStrokeGray(0.8); c.setLineWidth(0.4); c.rect(dx0 + (lw - iw * f) / 2, y - 7 * mm - ih * f, iw * f, ih * f)
        except Exception:
            L.wrap(c, 'Source image not embedded.', dx0, y - 10 * mm, lw, size=6.4, gray=0.4)
    else:
        L.key_plan(c, P, dx0, dy0 + 2 * mm, lw, box_h)
    yy = L.header(c, rx, y, rw, 'Sheet index')
    yy = L.table(c, rx, yy, ['Sheet', 'Title', 'Scale'], index, _fit_widths([0.2, 0.58, 0.22], rw), align=['l', 'l', 'l'], size=6.6, rh=5.0 * mm)
    yy -= 4 * mm
    yy = L.header(c, rx, yy, rw, 'Spaces')
    rows = [(f"S-{r['id']:02d}", _short(r.get('name', '—'), 30), FLOOR_CODE.get(r.get('floor', ''), '—'), f"{r['area']:.2f}")
            for r in P.rooms]
    if len(rows) > 12:
        rows = rows[:11] + [('…', f'{len(rows) - 11} more on A-102', '', '')]
    yy = L.table(c, rx, yy, ['Code', 'Space', 'Floor', 'm²'], rows, _fit_widths([0.16, 0.56, 0.12, 0.16], rw),
                 align=['l', 'l', 'l', 'r'], size=6.4, rh=4.5 * mm)
    yy -= 4 * mm
    yy = L.header(c, rx, yy, rw, 'What was read and what was assumed')
    for s_ in ('READ FROM THE IMAGE: walls, doors and windows, room names and printed dimensions, casework and fixtures, '
               'floor textures, furniture.',
               'MEASURED: the scale (' + meta.get('scaled_from', 'declared area.').rstrip('.') + '), every length and area.',
               'ASSUMED: heights (clear 2.45 m, doors 2.10 m), the electrical proposal, catalogue sizes of furniture and '
               'any fixture not visible in the image.',
               NOTE.upper() + '.'):
        yy = L.wrap(c, s_, rx, yy, rw, size=6.3, leading=7.9) - 2
    c.showPage()


def scale_texts(model, P):
    """Textos del rótulo y de los supuestos según de dónde salió la escala (v6)."""
    out = {}
    chk = model.get('area_check') or {}
    kind = chk.get('kind')
    out['area_label'] = {'private': 'Declared private area', 'gross': 'Declared built (gross) area'}.get(kind, 'Declared area')
    fit = model.get('scale_fit')
    s = model['m_per_px']
    terms = model.get('area_terms_px') or {}
    net = sum(r['area'] for r in P.rooms if not r.get('exterior'))
    if fit:
        sp = fit.get('spread')
        out['scaled_from'] = f"scaled from {fit['n']} printed room dimensions."
        a1 = (f"1. Scale fitted to {fit['n']} of the {fit['n_values']} room dimensions printed on the plan "
              f"({fit['n_rooms']} spaces; 90 % agree within ±{100 * (sp or 0):.1f} %)")
        if fit.get('aniso_before'):
            sx, sy = fit['aniso_before']
            a1 += f"; the image was stretched {100 * (sy / sx - 1):+.1f} % in height and was corrected"
        a1 += '.'
        if chk.get('area'):
            dev = chk.get('dev', 0.0)
            what = 'private area (net spaces + half of the partitions)' if kind == 'private' else 'built area (gross, with walls)'
            a1 += f" Declared {chk['area']:.1f} m² vs {what} measured at this scale: {100 * dev:+.1f} %."
        out['assume1'] = a1
    elif chk.get('area'):
        if kind == 'gross':
            out['scaled_from'] = 'scaled from the declared built (gross) area.'
            out['assume1'] = (f"1. Scale derived from the declared built area {chk['area']:.1f} m² (gross footprint with walls); "
                              f"net interior {net:.1f} m². No room dimensions printed on the plan to cross-check.")
        else:
            out['scaled_from'] = 'scaled from the declared private area.'
            out['assume1'] = (f"1. Scale derived from the declared private area {chk['area']:.1f} m² (net spaces + half of the partitions); "
                              f"net interior {net:.1f} m². No room dimensions printed on the plan to cross-check.")
    return out


def make_pdf(model, out_pdf, project='Marketing floor plan', unit='Unit', area=None, fin=None, area_kind=None, image=None):
    P = Plan(model, fin)
    from . import sections
    cuts = sections.pick_cuts(P) if fin else []
    net = sum(r['area'] for r in P.rooms if not r.get('exterior'))
    lay = plan_layout(P)
    meta = dict(project=project, unit=unit, area=area or model.get('area_m2'), net=net, P=P, plan_layout=lay,
                date=datetime.date.today().isoformat(), sheet_n=(6 + len(cuts)) if fin else 3)
    meta.update(scale_texts(model, P))
    _CUR['meta'] = meta; _CUR['P'] = P
    c = canvas.Canvas(out_pdf, pagesize=(PAGE_W, PAGE_H))
    c.setTitle(f'{project} — {unit} — architectural plan set')
    c.setAuthor('Zero Mouse Clicks')
    sc = f"1:{lay['scale']}"
    index = [('G-001', 'Cover sheet & index', 'n.t.s.'), ('A-101', 'Floor plan', sc), ('A-102', 'Areas & openings schedules', 'as noted')]
    if fin:
        index += [('A-103', 'Floor finishes plan', sc), ('A-104', 'Electrical layout (schematic)', sc), ('A-105', 'Furniture layout', sc)]
        index += [(f'A-20{i + 1}', cut['title'].title().replace('  ', ' '), f"1:{sections.section_scale(P, cut, SCALES)}") for i, cut in enumerate(cuts)]
    meta['sheet_i'] = 1
    sheet_cover(c, P, meta, index, image)
    meta['sheet_i'] = 2
    sheet_plan(c, P, meta, cuts=cuts)
    meta['sheet_i'] = 3
    sheet_schedules(c, P, meta, lay['scale'])
    if fin:
        meta['sheet_i'] = 4
        sheet_plan(c, P, meta, mode='finishes', title='FLOOR FINISHES PLAN', number='A-103')
        meta['sheet_i'] = 5
        sheet_plan(c, P, meta, mode='electrical', title='ELECTRICAL LAYOUT (SCHEMATIC)', number='A-104')
        meta['sheet_i'] = 6
        sheet_plan(c, P, meta, mode='furniture', title='FURNITURE LAYOUT', number='A-105')
        for i, cut in enumerate(cuts):
            meta['sheet_i'] = 7 + i
            meta['panel'] = None
            meta['keyplan'] = dict(cut=cut, cuts=[q for q in cuts if q is not cut])
            sections.sheet_section(c, P, meta, cut, f'A-20{i + 1}', _frame, F, FB, FRAME, TB_W, SCALES, _dim)
    c.save()
    return P
