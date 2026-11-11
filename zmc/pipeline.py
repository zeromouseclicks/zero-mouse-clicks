"""Imagen de folleto -> juego de planos (v5). Sin tokens: 3 modelos ONNX pequeños + reglas."""
import os
import cv2, numpy as np, onnxruntime as ort
from .vectorize import vectorize, SP, DR, WN, MS, AP, MU
from .vectorize5 import vectorize5
from . import finish, drawing, refine, debug as dbg, texts

_S = {}
def _sess():
    if not _S:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        opt = ort.SessionOptions(); opt.log_severity_level = 3
        load = lambda f: ort.InferenceSession(os.path.join(d, f), opt, providers=['CPUExecutionProvider'])
        _S['v4'] = load('unet_v4.onnx')
        _S['ens'] = [(load('unet_v2.onnx'), 640), (load('unet_v3.onnx'), 768)]
    return _S

def _probs(s, im, n):
    h, w = im.shape[:2]; x = np.full((n, n, 3), 255, np.uint8); x[:h, :w] = im
    z = s.run(None, {s.get_inputs()[0].name: x[None].astype(np.float32)})[0][0][:h, :w]
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def _fused(im):
    """Probabilidades: 5 clases (fondo, espacio, muro, puerta, ventana) promediando v2+v3+v4, y objetos de v4."""
    S = _sess()
    p4 = _probs(S['v4'], im, 768)
    q = p4[..., :5].copy(); q[..., 1] += p4[..., 5:].sum(-1)
    st = (q + sum(_probs(s, im, n) for s, n in S['ens'])) / 3.0
    return st, p4


def predict(im, tta=False):
    """Máscara de 8 clases. Lado máximo 640. Con tta=True promedia tres resoluciones (resultado más estable:
    una sola resolución cambia de plano según el tamaño de la imagen)."""
    st, p4 = _fused(im)
    if tta:
        h, w = im.shape[:2]
        n = 1
        for f in (0.85, 0.72):
            if max(h, w) * f < 352:
                continue
            sm = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            s2, q4 = _fused(sm)
            st = st + cv2.resize(s2, (w, h), interpolation=cv2.INTER_LINEAR)
            p4 = p4 + cv2.resize(q4, (w, h), interpolation=cv2.INTER_LINEAR)
            n += 1
        st, p4 = st / n, p4 / n
    m = np.argmax(st, -1).astype(np.uint8)
    ob = np.argmax(p4[..., 5:], -1).astype(np.uint8) + 5
    use = (m == 1) & (p4[..., 5:].sum(-1) > p4[..., 1]); m[use] = ob[use]
    return m

def _fit(im, side):
    r = side / max(im.shape[:2])
    return cv2.resize(im, None, fx=r, fy=r, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC)

def run(path, area, proj, unit, pdf, legacy=False, debug=None, side=None):
    """Imagen -> PDF. area: área privada en m² (None = leerla del folleto). Devuelve el modelo y un resumen."""
    im0 = cv2.imread(path)
    if im0 is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    info = {}
    cands = None
    if area is None:             # sin área: se busca en el folleto («Área privada: 47 m²»)
        cands = texts.find_areas(texts.read_words(im0, 1600) + texts.read_words(im0, 2200))
        info['areas_read'] = cands
        if not cands:
            raise ValueError('No private area given and none could be read from the image')
        area = cands[0][1]
    if legacy:
        im = _fit(im0, side or 448); m = predict(im)
        mod = vectorize(m.copy(), area); fin = finish.build(im, m, mod)
    else:
        imc = _fit(im0, 448); mc = predict(imc)
        mc, _ = refine.wall_color_filter(imc, mc)
        x0, y0, x1, y1 = refine.plan_bbox(mc)
        k = im0.shape[1] / imc.shape[1]
        crop = im0[int(y0 * k):int(np.ceil(y1 * k)), int(x0 * k):int(np.ceil(x1 * k))]
        fixed = side is not None
        side = side or 640
        im = _fit(crop, side); m = predict(im, tta=fixed)
        if not fixed:
            # resolución normalizada: ~55 px por metro, que es lo que el modelo vio al entrenar
            m1, _ = refine.wall_color_filter(im, m)
            ppm = 1.0 / vectorize5(m1.copy(), area, image=im)['m_per_px']
            if ppm > 64:
                side = int(np.clip(side * 55 / ppm, 416, 640))
                im = _fit(crop, side)
            m = predict(im, tta=True)
            info['ppm'] = round(ppm, 1)
        info['side'] = side
        m, like = refine.wall_color_filter(im, m, info)
        words = texts.read_words(crop, 2800)
        labs, dims = texts.parse(words)
        kk = im.shape[1] / crop.shape[1]
        for w in words:          # lo que el modelo «vio» debajo de un texto no es un objeto ni un vano
            if w['conf'] >= 60 and len(w['text']) >= 3 and w['h'] * kk < 0.08 * max(im.shape[:2]):
                a, b = int(w['x'] * kk), int(w['y'] * kk); c, d = int((w['x'] + w['w']) * kk) + 1, int((w['y'] + w['h']) * kk) + 1
                sub = m[b:d, a:c]; sub[np.isin(sub, (DR, WN, MS, AP, MU))] = SP
        mod = vectorize5(m.copy(), area, image=im)
        texts.apply(mod, labs, dims, kk)
        if cands and len(cands) > 1 and mod.get('scale_dims'):
            # varias áreas en el folleto (privada, construida…): la que mejor cuadra con las medidas impresas
            best = min(cands, key=lambda c: abs(np.log(np.sqrt(c[1] / mod['inner_px']) / mod['scale_dims'])))[1]
            if best != area:
                area = best
                mod = vectorize5(m.copy(), area, image=im); texts.apply(mod, labs, dims, kk)
        info['area'] = area
        fin = finish.build(im, m, mod)
        info.update(labels=[(l['name'], l['primary']) for l in labs], dims=[(q['a'], q['b']) for q in dims],
                    scale=dict(used=mod['m_per_px'], declared=mod.get('scale_declared'), dims=mod.get('scale_dims'), src=mod['scale_from']))
    P = drawing.make_pdf(mod, pdf, project=proj, unit=unit, area=area, fin=fin)
    if debug:
        cv2.imwrite(debug, dbg.render(im, m, mod, fin, 2), [cv2.IMWRITE_JPEG_QUALITY, 78])
    return dict(mod=mod, fin=fin, W=P.W, H=P.H, im=im, mask=m, info=info)
