"""Imagen de folleto -> juego de planos (v5). Sin tokens: 3 modelos ONNX pequeños + reglas.

El núcleo (`steps`) es un generador que pide al exterior dos cosas: correr un modelo
(('infer', nombre, x)) y leer textos (('ocr', [imágenes])). Así el mismo código corre en Python
(onnxruntime + Tesseract, `run`) y en el navegador (onnxruntime-web + tesseract.js, ver docs/app).
"""
import os
import cv2, numpy as np
from .vectorize import vectorize, SP, DR, WN, MS, AP, MU
from .vectorize5 import vectorize5
from . import finish, drawing, refine, debug as dbg, texts, calib

MODELS = {'v4': ('unet_v4.onnx', 768), 'v2': ('unet_v2.onnx', 640), 'v3': ('unet_v3.onnx', 768)}

_S = {}
def _sess(name):
    if name not in _S:
        import onnxruntime as ort
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        opt = ort.SessionOptions(); opt.log_severity_level = 3
        _S[name] = ort.InferenceSession(os.path.join(d, MODELS[name][0]), opt, providers=['CPUExecutionProvider'])
    return _S[name]

def local_backend(req):
    """Atiende las peticiones del generador en Python: modelos con onnxruntime, textos con Tesseract."""
    if req[0] == 'infer':
        s = _sess(req[1])
        return s.run(None, {s.get_inputs()[0].name: req[2][None].astype(np.float32)})[0][0]
    if req[0] == 'ocr':
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(req[1]) or 1) as ex:
            return list(ex.map(texts._tess, req[1]))
    raise ValueError(req[0])

def drive(gen, backend=local_backend):
    """Corre un generador de pasos atendiendo sus peticiones con `backend` (síncrono)."""
    try:
        req = next(gen)
        while True:
            req = gen.send(backend(req))
    except StopIteration as e:
        return e.value

def _probs(name, im):
    # modelos con forma dinámica: se rellena solo hasta el siguiente múltiplo de 32 (antes, siempre 640/768).
    # A 768 px un modelo pedía ~650 MB de memoria en onnxruntime-web; a 448, ~150 MB (el iPhone se cerraba).
    h, w = im.shape[:2]; n = max(64, -(-max(h, w) // 32) * 32)
    x = np.full((n, n, 3), 255, np.uint8); x[:h, :w] = im
    z = (yield ('infer', name, x))[:h, :w]
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def _fused(im):
    """Probabilidades: 5 clases (fondo, espacio, muro, puerta, ventana) promediando v2+v3+v4, y objetos de v4."""
    p4 = yield from _probs('v4', im)
    q = p4[..., :5].copy(); q[..., 1] += p4[..., 5:].sum(-1)
    p2 = yield from _probs('v2', im)
    p3 = yield from _probs('v3', im)
    st = (q + p2 + p3) / 3.0
    return st, p4


def predict(im, tta=False):
    """Versión síncrona de `predict_steps` (modelos locales)."""
    return drive(predict_steps(im, tta))

def predict_steps(im, tta=False):
    """Máscara de 8 clases. Lado máximo 640. Con tta=True promedia tres resoluciones (resultado más estable:
    una sola resolución cambia de plano según el tamaño de la imagen)."""
    st, p4 = yield from _fused(im)
    if tta:
        h, w = im.shape[:2]
        n = 1
        for f in (0.85, 0.72):
            if max(h, w) * f < 352:
                continue
            sm = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            s2, q4 = yield from _fused(sm)
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

def run(path, area, proj, unit, pdf, legacy=False, debug=None, side=None, area_kind=None):
    """Imagen -> PDF. area: área privada en m² (None = leerla del folleto). Devuelve el modelo y un resumen.
    `path` puede ser una ruta o una imagen BGR (numpy); `pdf` una ruta o un archivo abierto (BytesIO)."""
    im0 = cv2.imread(path) if isinstance(path, str) else path
    if im0 is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return drive(steps(im0, area, proj, unit, pdf, legacy, debug, side, area_kind=area_kind))

def steps(im0, area, proj, unit, pdf, legacy=False, debug=None, side=None, progress=None, area_kind=None, side_max=None):
    """Núcleo de `run` como generador de pasos (ver arriba). progress(etapa) avisa el avance; con datos
    (image, mask, box, mod, fin) entrega resultados intermedios para mostrarlos mientras se trabaja.
    area_kind: definición del área dada ('private' | 'gross' | None = no se sabe; se trata como privada).
    side_max: lado máximo de la imagen que leen los modelos (teléfonos: 512, para no agotar la memoria)."""
    say = progress or (lambda *_, **__: None)
    info = {}
    cands = []                   # áreas declaradas: (rango, m², tipo)
    if area is not None:
        cands = [(0, float(area), area_kind)]
    else:                        # sin área: se busca en el folleto («Área privada: 47 m²»)
        say('area')
        w1 = yield from texts.read_words_steps(im0, 1600)
        w2 = yield from texts.read_words_steps(im0, 2200)
        cands = texts.find_areas(w1 + w2)
        info['areas_read'] = cands
        if legacy and not cands:
            raise ValueError('No private area given and none could be read from the image')
    area0 = cands[0][1] if cands else None
    if legacy:
        area = area0
        im = _fit(im0, side or 448); m = yield from predict_steps(im)
        mod = vectorize(m.copy(), area); fin = finish.build(im, m, mod)
    else:
        say('locate')
        imc = _fit(im0, 448); mc = yield from predict_steps(imc)
        mc, _ = refine.wall_color_filter(imc, mc)
        x0, y0, x1, y1 = refine.plan_bbox(mc)
        say('found', image=imc, mask=mc, box=(x0, y0, x1, y1))
        k = im0.shape[1] / imc.shape[1]
        crop = im0[int(y0 * k):int(np.ceil(y1 * k)), int(x0 * k):int(np.ceil(x1 * k))]
        info['crop'] = (int(x0 * k), int(y0 * k), int(np.ceil(x1 * k)), int(np.ceil(y1 * k)))
        fixed = side is not None
        top = side_max or 640
        side = side or top
        say('read')
        im = _fit(crop, side); m = yield from predict_steps(im, tta=fixed)
        if not fixed:
            # resolución normalizada: ~55 px por metro, que es lo que el modelo vio al entrenar
            m1, _ = refine.wall_color_filter(im, m)
            say('seen', image=im, mask=m1)
            if area0:
                ppm = 1.0 / vectorize5(m1.copy(), area0, image=im)['m_per_px']
                if ppm > 64:
                    side = int(np.clip(side * 55 / ppm, min(416, top), top))
                    im = _fit(crop, side)
                info['ppm'] = round(ppm, 1)
            m = yield from predict_steps(im, tta=True)
        info['side'] = side
        info['kx'] = im.shape[1] / crop.shape[1]; info['ky'] = im.shape[0] / crop.shape[0]
        m, like = refine.wall_color_filter(im, m, info)
        say('seen', image=im, mask=m)
        say('texts')
        words = yield from texts.read_words_steps(crop, 2800)
        labs, dims = texts.parse(words)
        kk = im.shape[1] / crop.shape[1]
        for w in words:          # lo que el modelo «vio» debajo de un texto no es un objeto ni un vano
            if w['conf'] >= 60 and len(w['text']) >= 3 and w['h'] * kk < 0.08 * max(im.shape[:2]):
                a, b = int(w['x'] * kk), int(w['y'] * kk); c, d = int((w['x'] + w['w']) * kk) + 1, int((w['y'] + w['h']) * kk) + 1
                sub = m[b:d, a:c]; sub[np.isin(sub, (DR, WN, MS, AP, MU))] = SP
        say('vector')
        mod, im, m, labs, dims, sc = _scaled_model(im, m, labs, dims, kk, cands, like, typed=area is not None)
        info.update(sc)
        area = sc['area_used'] if 'area_used' in sc else area0
        info['area'] = area
        fin = finish.build(im, m, mod)
        say('drawn', image=im, mod=mod, fin=fin)
        info.update(labels=[(l['name'], l['primary']) for l in labs], dims=[(q['a'], q['b']) for q in dims],
                    scale=dict(used=mod['m_per_px'], declared=mod.get('scale_declared'), dims=mod.get('scale_dims'), src=mod['scale_from']))
    say('draw')
    P = drawing.make_pdf(mod, pdf, project=proj, unit=unit, area=area, fin=fin, area_kind=(mod.get('area_check') or {}).get('kind'))
    if debug:
        cv2.imwrite(debug, dbg.render(im, m, mod, fin, 2), [cv2.IMWRITE_JPEG_QUALITY, 78])
    return dict(mod=mod, fin=fin, W=P.W, H=P.H, im=im, mask=m, info=info)


def _apply_texts(mod, labs, dims, kk):
    texts.apply(mod, labs, dims, kk)
    mod['scale_dims'] = None


def _scaled_model(im, m, labs, dims, kk, cands, like=None, typed=False):
    """Vectoriza y fija la escala con todas las evidencias (v6):
    1) medidas impresas (si cuadran entre sí); 2) área declarada con su definición; 3) cruce entre ambas."""
    out = {}
    area0, kind0 = (cands[0][1], cands[0][2]) if cands else (None, None)
    mod = vectorize5(m.copy(), area0, image=im)
    _apply_texts(mod, labs, dims, kk)
    fit = calib.fit_dims(mod, m, dims, kk, texts._room_at, like) if dims else None
    if fit and fit['aniso']:
        # folleto estirado: se vuelve a muestrear para que el píxel mida lo mismo en X y en Y
        f = fit['sy'] / fit['sx']
        h, w = m.shape
        nh = int(round(h * f))
        im = cv2.resize(im, (w, nh), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (w, nh), interpolation=cv2.INTER_NEAREST)
        if like is not None:
            like = cv2.resize(like.astype(np.uint8), (w, nh), interpolation=cv2.INTER_NEAREST) > 0
        sy_ = lambda q: dict(q, y=q['y'] * f, h=q.get('h', 10) * f)
        labs = [sy_(q) for q in labs]; dims = [sy_(q) for q in dims]
        out['stretch_corrected'] = round(float(f), 4)
        mod = vectorize5(m.copy(), area0, image=im, scale=fit['sx'])
        _apply_texts(mod, labs, dims, kk)
        fit2 = calib.fit_dims(mod, m, dims, kk, texts._room_at, like)
        if fit2:
            fit2['aniso_before'] = (fit['sx'], fit['sy'])
            fit = fit2
    if fit and cands and fit['n_rooms'] < 3:
        # pocas medidas y un área que no cuadra con nada (±30 %): la lectura de medidas es dudosa
        terms0 = calib.area_terms(dict(mod, net_px=calib.regrow(mod)), im)
        s_ = float(np.sqrt(fit['sx'] * fit['sy']))
        if min(abs(calib.classify_area(c[1], s_, terms0)[1]) for c in cands) > 0.30:
            out['dims_rejected'] = dict(n=fit['n'], rooms=fit['n_rooms'])
            fit = None
    if fit:
        s = float(np.sqrt(fit['sx'] * fit['sy']))
        if abs(np.log(s / mod['s0'])) > 0.03:            # los vanos se clasifican en metros: rehacer con la escala buena
            mod = vectorize5(m.copy(), area0, image=im, scale=s)
            _apply_texts(mod, labs, dims, kk)
        mod['m_per_px'] = s
        mod['scale_from'] = f"{fit['n']} room dimensions printed on the plan"
        mod['scale_fit'] = fit
        out['dims_fit'] = dict(n=fit['n'], rooms=fit['n_rooms'], of=fit['n_values'], spread=fit['spread'], aniso=fit['aniso'])
    mod['net_px'] = calib.regrow(mod)
    terms = calib.area_terms(mod, im)
    mod['area_terms_px'] = terms
    if fit:
        chk = []
        for c in cands:
            k, e = calib.classify_area(c[1], mod['m_per_px'], terms)
            if c[2] and c[2] != k:                     # el rótulo manda sobre la definición; se informa el desvío
                k = c[2]; e = c[1] / (terms[k] * mod['m_per_px'] ** 2) - 1
            chk.append(dict(area=c[1], kind=k, dev=round(float(e), 4)))
        if not typed:
            # un número leído que no cuadra con nada (±15 %) no es el área del apartamento: se ignora
            chk = [q for q in chk if abs(q['dev']) <= 0.15]
        out['area_used'] = None
        if chk:
            best = min(chk, key=lambda q: (q['kind'] != 'private', abs(q['dev'])))
            mod['area_check'] = best
            out['area_used'] = best['area']
            mod['scale_declared'] = calib.scale_from_area(best['area'], best['kind'], terms)
        out['area_check'] = chk
    elif area0:
        kind = kind0 or 'private'
        s = calib.scale_from_area(area0, kind, terms)
        if abs(np.log(s / mod['s0'])) > 0.04:
            mod = vectorize5(m.copy(), area0, image=im, scale=s)
            _apply_texts(mod, labs, dims, kk)
            mod['net_px'] = calib.regrow(mod); terms = calib.area_terms(mod, im); mod['area_terms_px'] = terms
            s = calib.scale_from_area(area0, kind, terms)
        mod['m_per_px'] = s
        mod['scale_from'] = 'declared area'
        mod['scale_declared'] = s
        mod['area_check'] = dict(area=area0, kind=kind, dev=0.0)
        out['area_used'] = area0
    else:
        raise ValueError('No private area given and none could be read from the image, and the plan has no printed room dimensions')
    mod['area_m2'] = out['area_used'] if 'area_used' in out else area0
    return mod, im, m, labs, dims, out
