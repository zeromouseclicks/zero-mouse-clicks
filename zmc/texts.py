"""Lectura de los textos del folleto (OCR local con Tesseract; en la web: tesseract.js). Sin tokens.
- nombres de espacios (español / inglés) -> tipo de espacio
- medidas «2.69 x 2.93» -> escala real del dibujo
"""
import csv, io, os, re, subprocess, tempfile, unicodedata
import cv2
import numpy as np

KEYS = [  # (prefijos normalizados, tipo, nombre en el plano)
    (('habitacion', 'habitaci', 'alcoba', 'dormitorio', 'recamara', 'bedroom', 'cuarto', 'bed'), 'bed', 'BEDROOM'),
    (('bano', 'bafo', 'bafio', 'baho', 'bath', 'toilet', 'wc', 'aseo'), 'bath', 'BATHROOM'),
    (('cocina', 'kitchen', 'cocineta'), 'kitchen', 'KITCHEN'),
    (('sala', 'living', 'salon', 'estar', 'comedor', 'dining', 'family', 'great'), 'living', 'LIVING / DINING'),
    (('patio',), 'patio', 'PATIO'), (('terraza', 'terrace', 'deck'), 'patio', 'TERRACE'),
    (('balcon', 'balcony'), 'patio', 'BALCONY'), (('jardin', 'garden', 'yard'), 'patio', 'GARDEN'),
    (('ropas', 'lavanderia', 'laundry', 'labores', 'lavadero'), 'laundry', 'LAUNDRY'),
    (('estudio', 'study', 'office', 'oficina', 'den'), 'study', 'STUDY'),
    (('closet', 'vestier', 'vestidor', 'walk', 'wic'), 'closet', 'CLOSET'),
    (('hall', 'pasillo', 'corredor', 'foyer'), 'hall', 'HALL'),
    (('deposito', 'storage', 'bodega'), 'closet', 'STORAGE'),
]
PRIMARY = ('principal', 'master', 'primary', 'ppal')
NUM = re.compile(r'(?<!\d)(\d{1,2}[.,]\d{2})(?!\d)')


def _norm(s):
    s = unicodedata.normalize('NFD', s.lower())
    return re.sub(r'[^a-z0-9.,]', '', ''.join(c for c in s if unicodedata.category(c) != 'Mn'))


_WARNED = []


def available():
    """¿Hay Tesseract instalado? Sin él no se leen rótulos ni medidas (el plano sale igual, con nombres por reglas)."""
    import shutil
    ok = shutil.which('tesseract') is not None
    if not ok and not _WARNED:
        import sys
        print('zmc: Tesseract OCR not found; room labels and printed dimensions will not be read '
              '(install it to improve results, see README).', file=sys.stderr)
        _WARNED.append(1)
    return ok


def _tess(img):
    if not available():
        return []
    fd, p = tempfile.mkstemp(suffix='.png'); os.close(fd)
    cv2.imwrite(p, img)
    try:
        out = subprocess.run(['tesseract', p, '-', '--psm', '11', 'tsv'], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        out = ''
    os.remove(p)
    rows = []
    for r in csv.DictReader(io.StringIO(out), delimiter='\t', quoting=csv.QUOTE_NONE):
        try:
            t = (r.get('text') or '').strip(); c = float(r['conf'])
        except Exception:
            continue
        if t and c >= 40:
            rows.append((t, c, int(r['left']), int(r['top']), int(r['width']), int(r['height'])))
    return rows


def read_words(image, target=2400):
    """Palabras con caja, en coordenadas de `image`. Une varias pasadas (gris, invertido, trazos oscuros,
    trazos claros, umbral adaptativo): el texto de folleto va sobre pisos con textura."""
    from .pipeline import drive
    return drive(read_words_steps(image, target))


def read_words_steps(image, target=2400):
    """`read_words` como generador: pide ('ocr', [imágenes]) y recibe una lista de filas por imagen
    (texto, confianza, x, y, ancho, alto)."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    F = max(1.0, min(4.0, target / max(g.shape)))
    up = cv2.resize(g, None, fx=F, fy=F, interpolation=cv2.INTER_CUBIC)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(4.5 * F) | 1, int(4.5 * F) | 1))
    bh = cv2.normalize(cv2.morphologyEx(up, cv2.MORPH_BLACKHAT, k), None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.normalize(cv2.morphologyEx(up, cv2.MORPH_TOPHAT, k), None, 0, 255, cv2.NORM_MINMAX)
    ad = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 12)
    words = []
    for rows in (yield ('ocr', [up, 255 - up, 255 - bh, 255 - th, ad])):
        for t, c, x, y, w, h in rows:
            words.append(dict(text=t, conf=c, x=x / F, y=y / F, w=w / F, h=h / F))
    return words


def _key(n):
    for keys, typ, name in KEYS:
        for k in keys:
            if n == k or (len(k) >= 4 and n.startswith(k)) or (len(k) >= 7 and k[1:] in n):
                return typ, name
    return None


def parse(words):
    """-> (labels, dims). label: dict(type, name, x, y, h, primary); dim: dict(a, b, x, y)."""
    nums, labs, prim = [], [], []
    for w in words:
        n = _norm(w['text'])
        cy = w['y'] + w['h'] / 2
        found = NUM.findall(n)
        if found:
            cw = min(w['w'] / len(found), 2.7 * w['h'])       # cajas que se tragan el «x 2.20» vecino
            for i, f in enumerate(found):
                nums.append(dict(v=float(f.replace(',', '.')), conf=w['conf'], x=w['x'] + cw * (i + 0.5), y=cy, h=min(w['h'], 30)))
            continue
        if any(n.startswith(p) for p in PRIMARY):
            prim.append(dict(x0=w['x'], x1=w['x'] + w['w'], y=cy, h=min(w['h'], 30)))
            continue
        if len(n) < 2:
            continue
        kt = _key(n)
        if kt:
            labs.append(dict(type=kt[0], name=kt[1], x=w['x'] + w['w'] / 2, x0=w['x'], x1=w['x'] + w['w'], y=cy,
                             h=min(w['h'], 30), conf=w['conf'], primary=False))
    out = []
    for it in sorted(labs, key=lambda q: -q['conf']):
        if not any(o['type'] == it['type'] and abs(it['y'] - o['y']) < 1.2 * max(it['h'], o['h'])
                   and min(it['x1'], o['x1']) - max(it['x0'], o['x0']) > -1.0 * it['h'] for o in out):
            out.append(it)
    labs = out
    for p in prim:
        near = [l for l in labs if l['type'] == 'bed' and abs(p['y'] - l['y']) < 1.6 * p['h']
                and -1.5 * p['h'] < p['x0'] - l['x1'] < 2.5 * p['h']]
        if near:
            min(near, key=lambda l: abs(p['x0'] - l['x1']))['primary'] = True
        elif not any(l['primary'] and abs(l['y'] - p['y']) < 2 * p['h'] and abs(l['x'] - (p['x0'] + p['x1']) / 2) < 4 * p['h'] for l in labs):
            labs.append(dict(type='bed', name='BEDROOM', x=(p['x0'] + p['x1']) / 2, x0=p['x0'], x1=p['x1'], y=p['y'],
                             h=p['h'], conf=50, primary=True))
    out = []
    for it in sorted(nums, key=lambda q: -q['conf']):
        dup = False
        for o in out:
            dx, dy, hh = abs(it['x'] - o['x']), abs(it['y'] - o['y']), max(it['h'], o['h'])
            if dy < 0.7 * hh and (dx < 1.0 * hh or (it['v'] == o['v'] and dx < 2.2 * hh)):
                dup = True; break
        if not dup:
            out.append(it)
    nums = sorted(out, key=lambda q: (q['y'], q['x']))
    dims, used = [], set()
    for i, a in enumerate(nums):
        if i in used:
            continue
        best = None
        for j, b in enumerate(nums):
            if j == i or j in used:
                continue
            hh = max(a['h'], b['h'])
            if abs(a['y'] - b['y']) < 0.6 * hh and 1.5 * hh < b['x'] - a['x'] < 5.0 * hh:
                if best is None or b['x'] < nums[best]['x']:
                    best = j
        if best is not None:
            b = nums[best]; used |= {i, best}
            dims.append(dict(a=a['v'], b=b['v'], x=(a['x'] + b['x']) / 2, y=(a['y'] + b['y']) / 2, h=max(a['h'], b['h'])))
    return labs, dims


def find_areas(words):
    """Áreas declaradas en el folleto, como (no_es_privada, m²); la privada primero («Área Privada: 47 m²»)."""
    c = []
    ws = sorted(words, key=lambda w: (w['y'], w['x']))
    for i, w in enumerate(ws):
        n = _norm(w['text'])
        m = re.fullmatch(r'(\d{2,3}(?:[.,]\d{1,2})?)(m2|m\?|m)?', n) if n else None
        if not m:
            continue
        unit = m.group(2) or any(_norm(o['text']).startswith('m') and len(_norm(o['text'])) <= 2 and abs(o['y'] - w['y']) < w['h']
                                 and 0 < o['x'] - w['x'] < 4 * w['h'] + w['w'] for o in ws)
        if not unit:
            continue
        v = float(m.group(1).replace(',', '.'))
        if not 15 <= v <= 600:
            continue
        priv = any(_norm(o['text']).startswith(('priv', 'interior', 'net')) and 0 < w['y'] - o['y'] < 4.5 * w['h']
                   and abs(o['x'] - w['x']) < 14 * w['h'] for o in ws)
        c.append((not priv, v))
    return sorted(set(c))


def _room_at(lab, x, y, rad):
    h, w = lab.shape
    x, y = int(round(x)), int(round(y))
    if 0 <= x < w and 0 <= y < h and lab[y, x] > 0:
        return int(lab[y, x])
    sub = lab[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].ravel()
    sub = sub[sub > 0]
    return int(np.bincount(sub).argmax()) if sub.size else None


def apply(model, labels, dims, k):
    """Asigna rótulos y medidas a los espacios del modelo (k = px del modelo por px de la imagen leída)
    y, si las medidas son coherentes, fija la escala con ellas."""
    lab = model['room_labels']
    by_label = {r['label']: r for r in model['rooms']}
    rad = int(2.5 * model['thickness_px'])
    for l in labels:
        r = by_label.get(_room_at(lab, l['x'] * k, l['y'] * k, rad))
        if r is not None:
            r.setdefault('texts', []).append(dict(l, mx=l['x'] * k, my=l['y'] * k))
    vals = []
    for d in dims:
        r = by_label.get(_room_at(lab, d['x'] * k, d['y'] * k, rad))
        if r is None:
            continue
        r.setdefault('dims', []).append((d['a'], d['b']))
        w, h = r['bbox'][2], r['bbox'][3]
        opts = [(d['a'] / w, d['b'] / h), (d['b'] / w, d['a'] / h)]
        sw, sh = min(opts, key=lambda o: abs(np.log(o[0] / o[1])))
        r['dim_scale'] = (sw, sh)
        rect = r['area_px'] / float(max(1, w * h))          # solo espacios rectangulares con una sola medida
        if abs(np.log(sw / sh)) < 0.16 and not r.get('exterior') and rect >= 0.85 and len(r['dims']) == 1:
            vals += [sw, sh]
    model['scale_declared'] = model['m_per_px']
    model['scale_dims'] = float(np.median(vals)) if len(vals) >= 2 else None
    if model['scale_dims'] and not model.get('area_m2'):
        # sin área declarada: la escala sale de las medidas impresas
        model['m_per_px'] = model['scale_dims']
        model['scale_from'] = f'{len(vals) // 2} room dimension(s) printed on the plan'
    return model
