"""Modelo vectorial -> juego de planos PDF (A3 horizontal), solo reglas.

Lámina A-101: planta con ejes, cotas en cadena y totales, vanos, rótulos de espacio,
escala gráfica, rótulo y nota obligatoria.
Lámina A-102: cuadro de áreas, cuadro de puertas y ventanas, supuestos.
"""
import datetime
import numpy as np
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def _fonts():
    """Busca una fuente TTF con m² (DejaVu en Linux, la de matplotlib si está instalado, Arial en Windows/macOS)."""
    import os
    cands = [('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
             ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
             ('/Library/Fonts/Arial.ttf', '/Library/Fonts/Arial Bold.ttf'),
             ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf')]
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
        cands.insert(1, (os.path.join(d, 'DejaVuSans.ttf'), os.path.join(d, 'DejaVuSans-Bold.ttf')))
    except Exception:
        pass
    for reg, bold in cands:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont('ZSans', reg)); pdfmetrics.registerFont(TTFont('ZSans-Bold', bold))
                return 'ZSans', 'ZSans-Bold'
            except Exception:
                continue
    return 'Helvetica', 'Helvetica-Bold'


F, FB = _fonts()

PAGE_W, PAGE_H = landscape(A3)
FRAME = (20 * mm, 10 * mm, PAGE_W - 10 * mm, PAGE_H - 10 * mm)  # ISO 5457: margen de encuadernación 20 mm
TB_W = 95 * mm  # franja de rótulo a la derecha
NOTE = 'Drawn from marketing plan — verify dimensions on site'
SCALES = [20, 25, 50, 75, 100, 150, 200]


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
        self.rooms = [dict(id=r['id'], area=r['area_px'] * s * s, X=self.X(r['cx']), Y=self.Y(r['cy']))
                      for r in model['rooms']]
        self.t = model['thickness_px'] * s
        self.objs, self.elec, self.contours = [], [], {}
        if fin:
            import cv2
            for o in fin['objects']:
                q = dict(o); q.update(X0=self.X(o['x0']), Y0=self.Y(o['y1']), X1=self.X(o['x1']), Y1=self.Y(o['y0']))
                self.objs.append(q)
            self.elec = [dict(p, X=self.X(p['x']), Y=self.Y(p['y'])) for p in fin['electrical']]
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
    x0, y0, x1, y1 = FRAME
    c.setLineWidth(0.7 * mm / 1.0 * 0.5)
    c.rect(x0, y0, x1 - x0, y1 - y0)
    tb = x1 - TB_W
    c.setLineWidth(0.5)
    c.line(tb, y0, tb, y1)
    # marcas de centrado (ISO 5457)
    for (a, b, cc, d) in ((PAGE_W / 2, y0, PAGE_W / 2, y0 - 5 * mm), (PAGE_W / 2, y1, PAGE_W / 2, y1 + 5 * mm),
                          (x0, PAGE_H / 2, x0 - 5 * mm, PAGE_H / 2), (x1, PAGE_H / 2, x1 + 5 * mm, PAGE_H / 2)):
        c.line(a, b, cc, d)
    # rótulo (abajo a la derecha, ISO 7200 simplificado)
    rows = [('Title', title), ('Project', meta['project']), ('Unit', meta['unit']),
            ('Declared private area', f"{meta['area']:.1f} m²"), ('Scale', scale_txt),
            ('Date', meta['date']), ('Drawn by', 'Zero Mouse Clicks · rules engine'),
            ('Sheet', f"{number}  ({meta['sheet_i']}/{meta['sheet_n']})")]
    rh = 9 * mm
    yy = y0
    c.setFont(F, 6)
    for k, v in reversed(rows):
        c.line(tb, yy + rh, x1, yy + rh)
        c.setFont(F, 5.5); c.drawString(tb + 2 * mm, yy + rh - 3 * mm, k.upper())
        c.setFont(FB if k in ('Title', 'Sheet') else F, 9 if k == 'Title' else 8)
        c.drawString(tb + 2 * mm, yy + 2.3 * mm, v)
        yy += rh
    # marca
    c.setFont(FB, 13)
    c.drawString(tb + 4 * mm, y1 - 12 * mm, 'ZERO MOUSE CLICKS')
    c.setFont(F, 8)
    c.drawString(tb + 4 * mm, y1 - 17 * mm, 'Floor plan set from a marketing image')
    # nota obligatoria
    ny = yy + 6 * mm
    c.setFillGray(0.92); c.rect(tb + 3 * mm, ny, TB_W - 6 * mm, 30 * mm, stroke=0, fill=1); c.setFillGray(0)
    c.setFont(FB, 8.5)
    c.drawString(tb + 5 * mm, ny + 24 * mm, 'NOTE')
    c.setFont(F, 7.5)
    for i, line in enumerate(['Drawn from marketing plan —', 'verify dimensions on site.',
                              'Marketing plans are not to scale; this set is',
                              'scaled from the declared private area. Wall',
                              'thicknesses and openings are approximate.']):
        c.drawString(tb + 5 * mm, ny + (19 - 4 * i) * mm, line)
    return tb


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


def sheet_plan(c, P, meta, mode='plan', title='FLOOR PLAN', number='A-101', cuts=None):
    x0, y0, x1, y1 = FRAME
    avail_w = (x1 - TB_W) - x0 - 50 * mm
    avail_h = (y1 - y0) - 50 * mm
    scale = next(s for s in SCALES if P.W * 1000 / s * mm <= avail_w and P.H * 1000 / s * mm <= avail_h)
    k = 1000 / scale * mm  # puntos por metro
    ox = x0 + 30 * mm + (avail_w - P.W * k) / 2
    oy = y0 + 25 * mm + (avail_h - P.H * k) / 2
    tb = _frame(c, title, number, meta, f'1:{scale}  (A3)')

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
        _objects(c, P, T, k, light=(mode != 'plan'), furniture=(mode == 'plan'))
        if mode == 'electrical':
            _electrical(c, P, T, k)
    if cuts and mode == 'plan':
        from . import sections
        sections.draw_cut_marks(c, P, T, cuts, F, FB)
    # espacios
    def _tag(px, py, nm, sub):
        c.setFillGray(1); c.rect(px - len(nm) * 1.3 * mm / 1.6 - 1 * mm, py - (4 * mm if sub else 1 * mm), len(nm) * 2.6 * mm / 1.6 + 2 * mm,
                                  7.5 * mm if sub else 4.5 * mm, stroke=0, fill=1); c.setFillGray(0)
        c.setFont(FB, 6.5); c.drawCentredString(px, py + 1 * mm, nm)
        if sub:
            c.setFont(F, 6); c.drawCentredString(px, py - 2.5 * mm, sub)
    for r in P.rooms:
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
    # cotas: cadena entre ejes y total, arriba y a la izquierda
    if mode != 'plan':
        _legend(c, mode, tb)
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
    # escala gráfica y título de vista
    sx, sy = x0 + 12 * mm, y0 + 10 * mm
    c.setLineWidth(0.3)
    for i in range(5):
        c.setFillGray(0 if i % 2 == 0 else 1)
        c.rect(sx + i * k, sy, k, 1.5 * mm, fill=1)
    c.setFillGray(0); c.setFont(F, 6)
    for i in range(6):
        c.drawCentredString(sx + i * k, sy - 3 * mm, f'{i}')
    c.drawString(sx + 5 * k + 2 * mm, sy - 3 * mm, 'm')
    c.setFont(FB, 10)
    c.drawString(sx, sy + 6 * mm, f'{title}  —  1:{scale}')
    c.setFont(F, 7)
    sub = {'plan': 'Finished state: fixed casework and fixtures shown; loose furniture dashed (indicative only).',
           'finishes': 'Floor finishes read from the marketing image texture; confirm against the developer spec.',
           'electrical': 'SCHEMATIC PROPOSAL by rules — not in source plan. Verify with a licensed electrician / local code.'}[mode]
    c.drawString(sx, sy + 2.8 * mm, 'Dimensions in metres, to wall axes. ' + sub)
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


def _objects(c, P, T, k, light=False, furniture=True):
    """Mesones/closets, aparatos con símbolo y muebles insinuados."""
    g = 0.55 if light else 0
    for o in P.objs:
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


def _legend(c, mode, tb):
    x0, y0, x1, y1 = FRAME
    lx, ly = tb + 5 * mm, y1 - 30 * mm
    c.setFont(FB, 8.5); c.drawString(lx, ly, 'LEGEND')
    c.setFont(F, 7)
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


def sheet_schedules(c, P, meta, scale):
    x0, y0, x1, y1 = FRAME
    _frame(c, 'AREAS & OPENINGS SCHEDULES', 'A-102', meta, 'n.t.s.')

    def table(x, y, title, header, rows, widths):
        c.setFont(FB, 10); c.drawString(x, y, title)
        y -= 6 * mm
        rh = 6 * mm
        c.setLineWidth(0.4)
        c.setFillGray(0.88); c.rect(x, y - 1.8 * mm, sum(widths), rh, stroke=0, fill=1); c.setFillGray(0)
        c.setFont(FB, 7.5)
        xx = x
        for hd, w in zip(header, widths):
            c.drawString(xx + 1.5 * mm, y, hd); xx += w
        c.line(x, y - 1.8 * mm, x + sum(widths), y - 1.8 * mm)
        c.setFont(F, 7.5)
        for r in rows:
            y -= rh
            xx = x
            for v, w in zip(r, widths):
                c.drawString(xx + 1.5 * mm, y, str(v)); xx += w
            c.setLineWidth(0.15); c.line(x, y - 1.8 * mm, x + sum(widths), y - 1.8 * mm)
        return y

    inter = [r for r in P.rooms if not r.get('exterior')]
    exter = [r for r in P.rooms if r.get('exterior')]
    tot = sum(r['area'] for r in inter) or 1.0
    rows = [(f"S-{r['id']:02d}", r.get('name', '—'), FLOOR_CODE.get(r.get('floor', ''), '—'), f"{r['area']:.2f}",
             f"{100 * r['area'] / tot:.1f} %") for r in inter]
    rows.append(('TOTAL', 'net interior', '', f'{tot:.2f}', '100 %'))
    for r in exter:
        rows.append((f"S-{r['id']:02d}", r.get('name', '—') + ' (exterior)', FLOOR_CODE.get(r.get('floor', ''), '—'), f"{r['area']:.2f}", '—'))
    if exter:
        rows.append(('TOTAL', 'exterior', '', f"{sum(r['area'] for r in exter):.2f}", ''))
    y = table(x0 + 12 * mm, y1 - 18 * mm, 'AREA & FINISH SCHEDULE (m²)', ['Space', 'Use', 'Floor', 'Area', 'Share'],
              rows, [18 * mm, 40 * mm, 14 * mm, 20 * mm, 20 * mm])
    if P.objs:
        from collections import Counter
        cnt = Counter((o.get('fixture') or o.get('casework') or o['kind']) for o in P.objs if o['kind'] != 'furniture')
        rows2 = [(k.upper(), v) for k, v in sorted(cnt.items())]
        table(x0 + 238 * mm, y1 - 18 * mm, 'FIXTURES & CASEWORK', ['Item', 'Qty'], rows2,
              [38 * mm, 12 * mm])
    c.setFont(F, 7)
    c.drawString(x0 + 12 * mm, y - 8 * mm, f"Footprint (outer envelope): {P.W:.2f} x {P.H:.2f} m.  "
                                          f"Typical wall thickness ~{P.t * 100:.0f} cm.")
    ops = [(o['tag'], ('Sliding door' if o.get('style') == 'sliding' else 'Opening (no door)' if o.get('style') == 'opening' else 'Entry door' if o.get('entry') else 'Door') if o['kind'] == 'door' else 'Window', f"{o['width']:.2f}",
            '2.10' if o['kind'] == 'door' else '1.20 (assumed)') for o in P.ops]
    table(x0 + 132 * mm, y1 - 18 * mm, 'DOOR & WINDOW SCHEDULE', ['Tag', 'Type', 'Width (m)', 'Height (m)'],
          ops, [18 * mm, 22 * mm, 25 * mm, 30 * mm])
    # supuestos
    ax, ay = x0 + 12 * mm, y0 + 60 * mm
    c.setFont(FB, 9); c.drawString(ax, ay, 'ASSUMPTIONS')
    c.setFont(F, 7.5)
    for i, s in enumerate([
        f"1. Scale derived from {P.m.get('scale_from', 'declared area')}"
        + (f" (declared private area {meta['area']:.1f} m² = sum of net interior spaces)." if P.m.get('scale_from', 'declared area') == 'declared area' else '.')
        + (f" Cross-check with the room dimensions printed on the plan: {100 * (P.m['m_per_px'] / P.m['scale_dims'] - 1):+.1f} %." if P.m.get('scale_dims') else ''),
        '2. Walls, doors and windows read from the image by a small on-device model and squared by rules.',
        '3. Door heights 2.10 m and window heights 1.20 m are assumed; clear height 2.45 m assumed.',
        '4. Room names are read from the labels printed on the plan when present; otherwise inferred by rules.',
        '5. Electrical points (A-104) are a rule-based schematic proposal, not taken from the source plan.',
        '6. ' + NOTE + '.']):
        c.drawString(ax, ay - (6 + 5 * i) * mm, s)
    c.showPage()


def make_pdf(model, out_pdf, project='Marketing floor plan', unit='Unit', area=None, fin=None):
    P = Plan(model, fin)
    from . import sections
    cuts = sections.pick_cuts(P) if fin else []
    meta = dict(project=project, unit=unit, area=area or model['area_m2'],
                date=datetime.date.today().isoformat(), sheet_n=(4 + len(cuts)) if fin else 2)
    c = canvas.Canvas(out_pdf, pagesize=(PAGE_W, PAGE_H))
    c.setTitle(f'{project} — {unit} — floor plan set')
    c.setAuthor('Zero Mouse Clicks')
    meta['sheet_i'] = 1
    sc = sheet_plan(c, P, meta, cuts=cuts)
    meta['sheet_i'] = 2
    sheet_schedules(c, P, meta, sc)
    if fin:
        meta['sheet_i'] = 3
        sheet_plan(c, P, meta, mode='finishes', title='FLOOR FINISHES PLAN', number='A-103')
        meta['sheet_i'] = 4
        sheet_plan(c, P, meta, mode='electrical', title='ELECTRICAL LAYOUT (SCHEMATIC)', number='A-104')
        for i, cut in enumerate(cuts):
            meta['sheet_i'] = 5 + i
            sections.sheet_section(c, P, meta, cut, f'A-20{i + 1}', _frame, F, FB, FRAME, TB_W, SCALES, _dim)
    c.save()
    return P
