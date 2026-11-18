"""Folletos sintéticos con medidas impresas y verdad exacta, para el banco de medidas.

Cada planta sale de zmc.synth (mismas distribuciones, muebles y estilos) renderizada en alta resolución; encima
se escriben los nombres de los espacios y sus medidas libres («2.69 x 2.20» o «12'4" x 11'0"») como en un folleto.
La verdad (luces libres de cara a cara de muro, área neta, huella) se mide en el ráster supermuestreado.

Variantes:
  base      texto horizontal, 1600 px
  small     900 px (captura de pantalla / folleto web)
  vertical  en los espacios angostos la medida va girada 90°
  photo     foto de celular de un folleto impreso: perspectiva, giro, luz desigual, desenfoque, JPEG
  stretch   imagen estirada en X (5-12 %)

Uso:  python bench/measure/gen.py <carpeta> [n por variante] [semilla]
"""
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from zmc import synth as SY          # noqa: E402
import _render                       # noqa: E402

FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf',
    '/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf',
    '/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf',
    '/usr/share/fonts/opentype/inter/Inter-Regular.otf',
    '/usr/share/fonts/truetype/google-fonts/ofl/opensans/OpenSans[wdth,wght].ttf',
]
FONTS = [f for f in FONTS if os.path.exists(f)]
NAMES = {
    'co': dict(bed='Habitación', bath='Baño', kitchen='Cocina', living='Sala comedor', living2='Estudio',
               study='Estudio', laundry='Ropas', closet='Vestier'),
    'us': dict(bed='Bedroom', bath='Bath', kitchen='Kitchen', living='Living / Dining', living2='Den',
               study='Study', laundry='Laundry', closet='W.I.C.'),
}
PRINT = ('bed', 'bath', 'kitchen', 'living', 'living2', 'study')
VARIANTS = ('base', 'small', 'vertical', 'photo', 'stretch')


def _run(row, c0, c1):
    """Largo del tramo libre (False) que contiene la posición c0..c1 en una fila booleana de muro."""
    c = (c0 + c1) // 2
    if row[c]:
        return None
    l = c
    while l > 0 and not row[l - 1]:
        l -= 1
    r = c
    while r < len(row) - 1 and not row[r + 1]:
        r += 1
    return l, r + 1


def truth(hr):
    """Luces libres (m), área neta (m²) y rectángulo de cada espacio, medidos en el ráster supermuestreado."""
    wall = hr['wall'] | hr['door'] | hr['win']
    px = hr['px']
    out = []
    L = hr['layout']
    for i, (r, t) in enumerate(zip(L['rooms'], L['types'])):
        a, b = hr['P'](r[0], r[1]); c, d = hr['P'](r[2], r[3])
        xs, ys = [], []
        for f in (0.3, 0.5, 0.7):
            y = int(b + f * (d - b)); x = int(a + f * (c - a))
            q = _run(wall[y], a, c)
            if q:
                xs.append(q)
            q = _run(wall[:, x], b, d)
            if q:
                ys.append(q)
        if not xs or not ys:
            continue
        x0, x1 = int(np.median([q[0] for q in xs])), int(np.median([q[1] for q in xs]))
        y0, y1 = int(np.median([q[0] for q in ys])), int(np.median([q[1] for q in ys]))
        free = ~wall[y0:y1, x0:x1]
        out.append(dict(i=i, type=t, sx=(x1 - x0) / px, sy=(y1 - y0) / px, area=float(free.sum()) / px / px,
                        box_hr=[x0, y0, x1, y1]))
    union = np.zeros(wall.shape, np.uint8)
    for r in L['rooms']:
        p0, p1 = hr['P'](r[0], r[1]), hr['P'](r[2], r[3])
        cv2.rectangle(union, p0, p1, 255, -1)
    full = (union > 0) | hr['wall']
    ys_, xs_ = np.nonzero(full)
    gross = float(full.sum()) / px / px
    fp = ((xs_.max() - xs_.min() + 1) / px, (ys_.max() - ys_.min() + 1) / px)
    return out, gross, fp


def _fmt(v, unit, comma):
    if unit == 'm':
        return f'{v:.2f}'.replace('.', ',') if comma else f'{v:.2f}'
    inch = int(round(v / 0.0254))
    return f"{inch // 12}'{inch % 12}\""


def _text(im, xy, txt, font, fill, halo, angle=0):
    """Texto centrado en xy (px); halo = color del contorno o None; angle 0 / 90."""
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), txt, font=font, stroke_width=2 if halo else 0)
    w, h = r - l, b - t
    tile = Image.new('RGBA', (w + 8, h + 8), (0, 0, 0, 0))
    dt = ImageDraw.Draw(tile)
    if halo:
        dt.text((4 - l, 4 - t), txt, font=font, fill=fill, stroke_width=2, stroke_fill=halo)
    else:
        dt.text((4 - l, 4 - t), txt, font=font, fill=fill)
    if angle:
        tile = tile.rotate(angle, expand=True)
    im.paste(tile, (int(xy[0] - tile.size[0] / 2), int(xy[1] - tile.size[1] / 2)), tile)


def make(seed, variant='base'):
    """-> (imagen BGR, meta). La verdad va en metros; las coordenadas en px de la imagen final."""
    SY.RNG = np.random.default_rng(seed)
    rnd = np.random.default_rng(seed + 7919)
    style = ['co', 'co', 'us', 'line'][seed % 4]
    size = 900 if variant == 'small' else 1600
    hr = _render.render_hr(size, style)
    rooms, gross, fp = truth(hr)
    S = hr['S']
    img = np.clip(hr['img'], 0, 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    k = size / S                                   # px finales por px supermuestreado
    ppm = hr['px'] * k                              # px por metro en la imagen final
    unit = 'ft' if style == 'us' else 'm'
    lang = 'us' if style == 'us' else 'co'
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font_path = FONTS[seed % len(FONTS)]
    fh = float(rnd.uniform(0.20, 0.28)) * ppm      # alto de letra
    f_name = ImageFont.truetype(font_path, max(9, int(fh)))
    f_dim = ImageFont.truetype(font_path, max(8, int(fh * 0.92)))
    ink = tuple(int(v) for v in rnd.integers(20, 70, 3))
    halo = (250, 248, 244) if rnd.random() < 0.7 else None
    printed = []
    comma = rnd.random() < 0.25                    # coma decimal en todo el folleto
    nb = 0
    for r in rooms:
        if r['type'] == 'bed':
            nb += 1
        name = NAMES[lang][r['type']] + (f' {nb}' if r['type'] == 'bed' else '')
        x0, y0, x1, y1 = [v * k for v in r['box_hr']]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r['cx'], r['cy'] = cx, cy
        r['name'] = name
        do_dim = r['type'] in PRINT and rnd.random() < 0.85
        swap = rnd.random() < (0.15 if unit == 'ft' else 0.35)
        a, b = (r['sy'], r['sx']) if swap else (r['sx'], r['sy'])
        dim_txt = f"{_fmt(a, unit, comma)} x {_fmt(b, unit, comma)}"
        wn = ImageDraw.Draw(pil).textlength(name, font=f_name)
        wd = ImageDraw.Draw(pil).textlength(dim_txt, font=f_dim)
        narrow = max(wn, wd) > 0.85 * (x1 - x0)
        if narrow and (y1 - y0) < 1.3 * (x1 - x0):
            continue                                 # no cabe: el folleto no rotula este espacio
        vert = narrow or (variant == 'vertical' and (x1 - x0) < (y1 - y0) and rnd.random() < 0.7)
        if vert:
            _text(pil, (cx - 0.6 * fh, cy), name, f_name, ink, halo, 90)
            if do_dim:
                _text(pil, (cx + 0.6 * fh, cy), dim_txt, f_dim, ink, halo, 90)
        else:
            _text(pil, (cx, cy - (0.6 * fh if do_dim else 0)), name, f_name, ink, halo)
            if do_dim:
                _text(pil, (cx, cy + 0.6 * fh), dim_txt, f_dim, ink, halo)
        r['label_vertical'] = bool(vert)
        if do_dim:
            r['printed'] = [round(a, 2), round(b, 2)] if unit == 'm' else [a, b]
            r['printed_txt'] = dim_txt
            printed.append(r['i'])
    area_txt = None
    if rnd.random() < 0.5:
        area_txt = ('Área construida: ' if lang == 'co' else 'Total: ') + \
            (f'{gross:.0f} m²' if unit == 'm' else f'{gross / 0.09290304:.0f} sq ft')
        _text(pil, (size * 0.5, size * 0.035), area_txt, f_name, ink, None)
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    Hm = np.eye(3)
    if variant == 'stretch':
        f = float(rnd.uniform(1.05, 1.12))
        img = cv2.resize(img, (int(round(size * f)), size), interpolation=cv2.INTER_AREA)
        Hm = np.diag([f, 1.0, 1.0])
    if variant == 'photo':
        img, Hm = photo(img, rnd)
    for r in rooms:
        p = Hm @ np.array([r['cx'], r['cy'], 1.0])
        r['cx_img'], r['cy_img'] = float(p[0] / p[2]), float(p[1] / p[2])
        r.pop('box_hr', None)
    meta = dict(seed=seed, variant=variant, style=style, unit=unit, ppm=ppm, gross=gross, footprint=fp,
                area_txt=area_txt, rooms=rooms, H=Hm.tolist(), font=os.path.basename(font_path))
    return img, meta


def photo(img, rnd):
    """Foto de celular de un folleto impreso: hoja sobre una mesa, perspectiva, giro, luz y desenfoque."""
    h, w = img.shape[:2]
    pad = int(0.12 * max(h, w))
    W, H = w + 2 * pad, h + 2 * pad
    tilt = float(rnd.uniform(0.02, 0.07))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda: float(rnd.uniform(-tilt, tilt)) * w
    dst = np.float32([[pad + j(), pad + j()], [pad + w + j(), pad + j()], [pad + w + j(), pad + h + j()], [pad + j(), pad + h + j()]])
    ang = float(rnd.uniform(-4, 4))
    R = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    R3 = np.vstack([R, [0, 0, 1]])
    Hm = R3 @ cv2.getPerspectiveTransform(src, dst)
    table = np.full((H, W, 3), [int(v) for v in rnd.integers(60, 140, 3)], np.uint8)
    out = cv2.warpPerspective(img, Hm, (W, H), dst=table, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1.0 + float(rnd.uniform(-0.18, 0.05)) * (xx / W) + float(rnd.uniform(-0.18, 0.05)) * (yy / H)
    out = np.clip(out.astype(np.float32) * g[..., None] * float(rnd.uniform(0.85, 1.0)), 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), float(rnd.uniform(0.6, 1.4)))
    noise = rnd.normal(0, 3.0, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out = cv2.imdecode(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, int(rnd.integers(70, 90))])[1], 1)
    return out, Hm


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    os.makedirs(out, exist_ok=True)
    for v in VARIANTS:
        for i in range(n):
            seed = seed0 + i
            img, meta = make(seed, v)
            name = f'{v}_{seed}'
            cv2.imwrite(os.path.join(out, name + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            json.dump(meta, open(os.path.join(out, name + '.json'), 'w'), indent=1)
            print(name, img.shape, len([r for r in meta['rooms'] if 'printed' in r]), 'printed', flush=True)


if __name__ == '__main__':
    main()
