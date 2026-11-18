"""Formato de lámina (v0.3): marco ISO 5457 con zonas, columna de rótulo y piezas de diagramación.

Columna de rótulo (derecha, de arriba abajo): marca · panel propio de la lámina · plano llave · notas
generales · revisiones · rótulo (proyecto, unidad, áreas, escala, fecha, título y número grande).
El área de dibujo queda a la izquierda; drawing_area() la entrega en puntos.
"""
import math
import textwrap

from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
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
TB_W = 88 * mm                                                   # columna de rótulo
ACC = (0.10, 0.27, 0.47)                                         # acento (azul tinta), solo en detalles
MUTE = 0.42
RULE = 0.72
NOTE = 'Drawn from marketing plan — verify dimensions on site'


def drawing_area():
    x0, y0, x1, y1 = FRAME
    return (x0 + 7 * mm, y0 + 22 * mm, x1 - TB_W - 7 * mm, y1 - 7 * mm)


# ------------------------------------------------------------------------------------------ piezas básicas
def label(c, x, y, txt, size=5.2, gray=MUTE, font=None):
    c.saveState(); c.setFillGray(gray); c.setFont(font or F, size); c.drawString(x, y, txt.upper()); c.restoreState()


def header(c, x, y, w, txt, size=7.2):
    """Título de bloque: texto en negrita y una línea fina debajo."""
    c.saveState()
    c.setFont(FB, size); c.setFillColorRGB(*ACC); c.drawString(x, y, txt.upper())
    c.setStrokeGray(0.15); c.setLineWidth(0.6); c.line(x, y - 1.6 * mm, x + w, y - 1.6 * mm)
    c.restoreState()
    return y - 5.2 * mm


def wrap(c, text, x, y, width, size=6.6, leading=None, font=None, gray=0.12, max_lines=None):
    """Texto en párrafo dentro de `width` puntos; devuelve la y de la línea siguiente."""
    font = font or F
    leading = leading or size * 1.32
    c.saveState(); c.setFont(font, size); c.setFillGray(gray)
    n = 0
    for para in text.split('\n'):
        words, line = para.split(), ''
        for wd in words:
            t = (line + ' ' + wd).strip()
            if c.stringWidth(t, font, size) <= width:
                line = t
            else:
                c.drawString(x, y, line); y -= leading; n += 1; line = wd
                if max_lines and n >= max_lines:
                    c.restoreState(); return y
        if line:
            c.drawString(x, y, line); y -= leading; n += 1
            if max_lines and n >= max_lines:
                break
    c.restoreState()
    return y


def table(c, x, y, header_row, rows, widths, size=6.4, rh=4.3 * mm, align=None, total=None, bars=None, max_h=None, more='on A-102'):
    """Cuadro simple: encabezado gris, filas con línea fina, fila de total en negrita.
    align: lista 'l'/'r'/'c' por columna. bars: índice de columna con fracción 0..1 para dibujar una barra."""
    align = align or ['l'] * len(widths)
    W = sum(widths)
    if max_h is not None:
        n_fit = int((max_h - rh - (rh if total else 0) - 1.5 * mm) // rh)
        if n_fit < 1:
            return y
        if len(rows) > n_fit:
            extra = len(rows) - n_fit + 1
            rows = list(rows[:n_fit - 1]) + [tuple(['…', f'{extra} more {more}'] + [''] * (len(widths) - 2))]
    c.saveState()
    c.setFillGray(0.90); c.rect(x, y - rh + 1.2 * mm, W, rh, stroke=0, fill=1)
    c.setFillGray(0.1); c.setFont(FB, size - 0.6)

    def cell(xx, w, t, a, yy):
        t = str(t)
        if a == 'r':
            c.drawRightString(xx + w - 1.4 * mm, yy, t)
        elif a == 'c':
            c.drawCentredString(xx + w / 2, yy, t)
        else:
            c.drawString(xx + 1.4 * mm, yy, t)
    xx = x
    for h, w, a in zip(header_row, widths, align):
        cell(xx, w, h.upper(), a, y - rh + 2.6 * mm); xx += w
    y -= rh
    c.setStrokeGray(0.2); c.setLineWidth(0.5); c.line(x, y + 1.2 * mm, x + W, y + 1.2 * mm)
    c.setFont(F, size)
    for i, r in enumerate(rows):
        xx = x
        for j, (v, w, a) in enumerate(zip(r, widths, align)):
            if bars is not None and j == bars and isinstance(v, (tuple, list)):
                frac, txt = v
                c.setFillColorRGB(0.80, 0.86, 0.93)
                c.rect(xx + 1.2 * mm, y - rh + 2.0 * mm, max(0.2, (w - 2.4 * mm) * min(1, max(0, frac))), rh - 1.6 * mm, stroke=0, fill=1)
                c.setFillGray(0.1); cell(xx, w, txt, a, y - rh + 2.6 * mm)
            else:
                c.setFillGray(0.1); cell(xx, w, v, a, y - rh + 2.6 * mm)
            xx += w
        y -= rh
        c.setStrokeGray(0.82); c.setLineWidth(0.25); c.line(x, y + 1.2 * mm, x + W, y + 1.2 * mm)
    if total:
        c.setFont(FB, size)
        xx = x
        for v, w, a in zip(total, widths, align):
            c.setFillGray(0.1); cell(xx, w, v, a, y - rh + 2.6 * mm); xx += w
        y -= rh
        c.setStrokeGray(0.2); c.setLineWidth(0.5); c.line(x, y + 1.2 * mm, x + W, y + 1.2 * mm)
    c.restoreState()
    return y - 1.5 * mm


def logo(c, x, y, s=6.4 * mm):
    """Marca: un recinto con un espacio en la esquina (planta mínima)."""
    c.saveState()
    c.setStrokeGray(0.05); c.setLineWidth(1.1)
    c.rect(x, y, s, s, stroke=1, fill=0)
    c.setFillGray(0.05)
    c.rect(x, y + s * 0.52, s * 0.48, s * 0.48, stroke=0, fill=1)
    c.restoreState()


def view_title(c, x, y, num, title, scale_txt, sub=None):
    """Título de vista: burbuja con número, título en negrita sobre una línea gruesa, escala debajo."""
    c.saveState()
    r = 3.6 * mm
    c.setLineWidth(0.6); c.setStrokeGray(0.05); c.circle(x + r, y + r * 0.6, r)
    c.setFont(FB, 9); c.drawCentredString(x + r, y + r * 0.6 - 3.1, str(num))
    tx = x + 2 * r + 3 * mm
    c.setFont(FB, 10.5); c.drawString(tx, y + 2.2 * mm, title)
    tw = c.stringWidth(title, FB, 10.5)
    c.setLineWidth(1.4); c.line(tx, y + 0.6 * mm, tx + max(tw, 45 * mm) + 12 * mm, y + 0.6 * mm)
    c.setLineWidth(0.4); c.line(tx, y - 0.3 * mm, tx + max(tw, 45 * mm) + 12 * mm, y - 0.3 * mm)
    c.setFont(F, 7); c.setFillGray(0.2); c.drawString(tx, y - 3.8 * mm, scale_txt)
    if sub:
        c.setFillGray(MUTE); c.setFont(F, 6.3); c.drawString(tx, y - 7.3 * mm, sub)
    c.restoreState()


def scale_bar(c, x, y, k, n=5, label_m=True):
    """Escala gráfica de n metros (k = puntos por metro)."""
    c.saveState(); c.setLineWidth(0.3)
    step = 1 if n * k < 160 * mm else 2
    for i in range(0, n, step):
        c.setFillGray(0.1 if (i // step) % 2 == 0 else 1)
        c.rect(x + i * k, y, step * k, 1.4 * mm, fill=1)
    c.setFillGray(0.1); c.setFont(F, 5.6)
    for i in range(0, n + 1, step):
        c.drawCentredString(x + i * k, y - 2.8 * mm, str(i))
    if label_m:
        c.drawString(x + n * k + 1.8 * mm, y - 2.8 * mm, 'm')
    c.restoreState()


# ------------------------------------------------------------------------------------------ plano llave
def key_plan(c, P, x, y, w, h, cut=None, cuts=None, codes=False, fills=None):
    """Planta mínima de la unidad dentro de (x, y, w, h). cut = corte a resaltar; codes = códigos de espacio."""
    if P is None or P.W <= 0 or P.H <= 0:
        return
    k = min(w / P.W, h / P.H)
    ox = x + (w - P.W * k) / 2
    oy = y + (h - P.H * k) / 2

    def T(xx, yy):
        return ox + xx * k, oy + yy * k
    c.saveState()
    for r in P.rooms:
        poly = P.contours.get(r['id'])
        if not poly:
            continue
        path = c.beginPath()
        px, py = T(*poly[0]); path.moveTo(px, py)
        for pt in poly[1:]:
            px, py = T(*pt); path.lineTo(px, py)
        path.close()
        col = (fills or {}).get(r['id'])
        if col:
            c.setFillColorRGB(*col)
        else:
            c.setFillGray(0.86 if r.get('exterior') else 0.93)
        c.drawPath(path, stroke=0, fill=1)
    c.setFillGray(0.15)
    for o, a, b, cc, d in P.walls:
        p, q = T(a, b), T(cc, d)
        c.rect(p[0], p[1], max(0.4, q[0] - p[0]), max(0.4, q[1] - p[1]), stroke=0, fill=1)
    if codes:
        c.setFont(FB, 5.4 if k > 6 else 4.6)
        for r in P.rooms:
            px, py = T(r['X'], r['Y'])
            c.setFillGray(0.1); c.drawCentredString(px, py - 1.6, f"S-{r['id']:02d}")
    for ct in ([cut] if cut else []) + list(cuts or []):
        c.setStrokeColorRGB(*ACC); c.setLineWidth(1.0 if ct is cut else 0.5)
        if ct['axis'] == 'y':
            a, b = T(-0.3, ct['c']), T(P.W + 0.3, ct['c'])
        else:
            a, b = T(ct['c'], -0.3), T(ct['c'], P.H + 0.3)
        c.line(a[0], a[1], b[0], b[1])
        c.setFillColorRGB(*ACC); c.setFont(FB, 5.5)
        for p in (a, b):
            c.circle(p[0], p[1], 1.8 * mm, stroke=0, fill=1)
            c.setFillGray(1); c.drawCentredString(p[0], p[1] - 1.9, ct['name']); c.setFillColorRGB(*ACC)
    c.restoreState()


# ------------------------------------------------------------------------------------------ marco y rótulo
def border(c):
    x0, y0, x1, y1 = FRAME
    c.saveState()
    c.setStrokeGray(0); c.setLineWidth(1.4)
    c.rect(x0, y0, x1 - x0, y1 - y0)
    # zonas de referencia (ISO 5457): 8 columnas y 6 filas en el margen
    c.setLineWidth(0.35); c.setFont(F, 5.5); c.setFillGray(0.35)
    nx, ny = 8, 6
    for i in range(nx + 1):
        xx = x0 + (x1 - x0) * i / nx
        c.line(xx, y1, xx, y1 + 3 * mm); c.line(xx, y0, xx, y0 - 3 * mm)
        if i < nx:
            xm = x0 + (x1 - x0) * (i + 0.5) / nx
            c.drawCentredString(xm, y1 + 2.6 * mm, str(i + 1)); c.drawCentredString(xm, y0 - 4.6 * mm, str(i + 1))
    for j in range(ny + 1):
        yy = y0 + (y1 - y0) * j / ny
        c.line(x0, yy, x0 - 3 * mm, yy); c.line(x1, yy, x1 + 3 * mm, yy)
        if j < ny:
            ym = y0 + (y1 - y0) * (j + 0.5) / ny - 2
            letter = 'ABCDEF'[ny - 1 - j]
            c.drawCentredString(x0 - 4.5 * mm, ym, letter); c.drawCentredString(x1 + 4.5 * mm, ym, letter)
    # marcas de centrado
    c.setLineWidth(0.7); c.setStrokeGray(0)
    for (a, b, cc, d) in ((PAGE_W / 2, y0, PAGE_W / 2, y0 - 5 * mm), (PAGE_W / 2, y1, PAGE_W / 2, y1 + 5 * mm),
                          (x0, PAGE_H / 2, x0 - 5 * mm, PAGE_H / 2), (x1, PAGE_H / 2, x1 + 5 * mm, PAGE_H / 2)):
        c.line(a, b, cc, d)
    c.restoreState()


def column(c, title, number, meta, scale_txt):
    """Columna de rótulo. meta['panel'](c, x, y_top, w, h) dibuja el panel propio de la lámina;
    meta['keyplan'] = dict(cut=..., cuts=...) o None para omitirlo. Devuelve la x del borde izquierdo."""
    x0, y0, x1, y1 = FRAME
    tb = x1 - TB_W
    pad = 3.6 * mm
    W = TB_W - 2 * pad
    c.saveState()
    c.setStrokeGray(0); c.setLineWidth(0.9); c.line(tb, y0, tb, y1)

    # 1. marca
    top = y1
    logo(c, tb + pad, top - 13.2 * mm)
    c.setFont(FB, 11.5); c.setFillGray(0.05); c.drawString(tb + pad + 9 * mm, top - 9.0 * mm, 'ZERO MOUSE CLICKS')
    c.setFont(F, 6.4); c.setFillGray(MUTE); c.drawString(tb + pad + 9 * mm, top - 12.9 * mm, 'Architectural plan set from a marketing image')
    top -= 17 * mm
    c.setLineWidth(0.5); c.setStrokeGray(0); c.line(tb, top, x1, top)

    # alturas fijas de abajo hacia arriba
    h_title, h_rev, h_notes, h_key = 61 * mm, 13 * mm, 29 * mm, (40 * mm if meta.get('keyplan', {}) is not None else 0)
    y_title = y0 + h_title
    y_rev = y_title + h_rev
    y_notes = y_rev + h_notes
    y_key = y_notes + h_key

    # 2. panel propio
    panel = meta.get('panel')
    if panel:
        c.saveState()
        panel(c, tb + pad, top - 5.5 * mm, W, top - 5.5 * mm - (y_key + 3 * mm))
        c.restoreState()
    c.setLineWidth(0.5); c.setStrokeGray(0)
    if h_key:
        c.line(tb, y_key, x1, y_key)
        header(c, tb + pad, y_key - 4.6 * mm, W, 'Key plan')
        kp = meta.get('keyplan') or {}
        key_plan(c, meta.get('P'), tb + pad, y_notes + 3 * mm, W, h_key - 11 * mm, cut=kp.get('cut'), cuts=kp.get('cuts'))

    # 3. notas generales
    c.setLineWidth(0.5); c.setStrokeGray(0); c.line(tb, y_notes, x1, y_notes)
    yy = header(c, tb + pad, y_notes - 4.6 * mm, W, 'General notes')
    notes = [f'1. {NOTE}.',
             '2. Marketing plans are not to scale; this set is ' + meta.get('scaled_from', 'scaled from the declared private area.'),
             '3. Dimensions in metres, to wall axes unless noted. Wall thicknesses and openings are approximate.']
    for n_ in notes:
        yy = wrap(c, n_, tb + pad, yy, W, size=5.9, leading=7.0) - 0.6

    # 4. revisiones
    c.setLineWidth(0.5); c.setStrokeGray(0); c.line(tb, y_rev, x1, y_rev)
    label(c, tb + pad, y_rev - 3.4 * mm, 'Rev'); label(c, tb + pad + 9 * mm, y_rev - 3.4 * mm, 'Date')
    label(c, tb + pad + 27 * mm, y_rev - 3.4 * mm, 'Description')
    c.setStrokeGray(RULE); c.setLineWidth(0.25); c.line(tb + pad, y_rev - 4.6 * mm, x1 - pad, y_rev - 4.6 * mm)
    c.setFont(F, 6.3); c.setFillGray(0.1)
    c.drawString(tb + pad, y_rev - 8.4 * mm, '0'); c.drawString(tb + pad + 9 * mm, y_rev - 8.4 * mm, meta['date'])
    c.drawString(tb + pad + 27 * mm, y_rev - 8.4 * mm, 'First issue, read from the marketing image')

    # 5. rótulo
    c.setLineWidth(0.9); c.setStrokeGray(0); c.line(tb, y_title, x1, y_title)
    rows = [  # (alto, [(fracción de ancho, etiqueta, valor, fuente, tamaño)])
        (9.5 * mm, [(1.0, 'Project', meta['project'], FB, 8.4)]),
        (8 * mm, [(1.0, 'Unit / type', meta['unit'], F, 7.6)]),
        (8 * mm, [(0.5, meta.get('area_label', 'Declared area'), f"{meta['area']:.1f} m²" if meta.get('area') else '—', F, 7.4),
                  (0.5, 'Net interior (measured)', f"{meta['net']:.1f} m²" if meta.get('net') else '—', F, 7.4)]),
        (8 * mm, [(0.5, 'Scale', scale_txt, F, 7.4), (0.5, 'Date', meta['date'], F, 7.4)]),
        (11.5 * mm, [(1.0, 'Sheet title', title, FB, 9.6)]),
        (16 * mm, None),
    ]
    yy = y_title
    for h, cells in rows:
        yb = yy - h
        if cells is None:   # número de lámina, grande
            c.setFillGray(0.05); c.setFont(FB, 22); c.drawString(tb + pad, yb + 4.2 * mm, number)
            nw = c.stringWidth(number, FB, 22)
            c.setFont(F, 7); c.setFillGray(MUTE)
            c.drawString(tb + pad + nw + 4 * mm, yb + 9.4 * mm, f"SHEET {meta['sheet_i']} OF {meta['sheet_n']}")
            c.drawString(tb + pad + nw + 4 * mm, yb + 5.2 * mm, 'A3 · 420 × 297 mm')
            label(c, tb + pad, yb + h - 3.2 * mm, 'Sheet no.')
        else:
            xx = tb
            for frac, lb, val, font, size in cells:
                cw = TB_W * frac
                label(c, xx + pad, yb + h - 3.2 * mm, lb)
                c.setFillGray(0.05); c.setFont(font, size)
                v = str(val)
                while c.stringWidth(v, font, size) > cw - 2 * pad and len(v) > 4:
                    v = v[:-2]
                c.drawString(xx + pad, yb + 1.9 * mm, v)
                if xx > tb:
                    c.setStrokeGray(0); c.setLineWidth(0.3); c.line(xx, yb, xx, yy)
                xx += cw
        c.setStrokeGray(0); c.setLineWidth(0.3)
        if yb > y0 + 0.1:
            c.line(tb, yb, x1, yb)
        yy = yb
    c.restoreState()
    return tb


def frame(c, title, number, meta, scale_txt):
    border(c)
    return column(c, title, number, meta, scale_txt)
