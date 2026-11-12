"""Zero Mouse Clicks en el navegador (Pyodide). Mismo pipeline v5 que la CLI.

Los modelos corren con onnxruntime-web y los textos con tesseract.js; este módulo atiende
las peticiones del generador `zmc.pipeline.steps` llamando a esas funciones de JavaScript.
"""
import io
import json

import cv2
import numpy as np
from pyodide.ffi import to_js

from zmc import pipeline


class NoArea(Exception):
    pass


INK, DOOR, WIN, FIX, DARK = (255, 208, 159), (120, 200, 255), (200, 255, 160), (255, 170, 190), (10, 7, 5)   # BGR


def _preview(image, mask=None, box=None, mod=None, fin=None):
    """Imagen (JPG) de un resultado intermedio: lo que ven los modelos o el dibujo ya vectorizado."""
    f = 640.0 / max(image.shape[:2])
    h, w = int(round(image.shape[0] * f)), int(round(image.shape[1] * f))
    if mod is None:
        out = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) * 0.16 + np.array(DARK, np.float32) * 0.84
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out[m == 1] += np.array(INK, np.float32) * 0.10
        for k, c in ((2, INK), (3, DOOR), (4, WIN)):
            out[m == k] = c
        obj = m >= 5
        out[obj] = out[obj] * 0.4 + np.array(FIX, np.float32) * 0.6
        out = np.clip(out, 0, 255).astype(np.uint8)
        if box is not None:
            x0, y0, x1, y1 = [int(round(v * f)) for v in box]
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1, cv2.LINE_AA)
    else:
        out = np.full((h, w, 3), DARK, np.uint8)
        q = lambda v: int(round(v * f))
        for o, x0, y0, x1, y1 in mod['walls']:
            cv2.rectangle(out, (q(x0), q(y0)), (q(x1) - 1, q(y1) - 1), INK, -1)
        for op in mod['openings']:
            cv2.rectangle(out, (q(op['x0']), q(op['y0'])), (q(op['x1']) - 1, q(op['y1']) - 1), DOOR if op['kind'] == 'door' else WIN, -1)
        for r in mod['rooms']:
            rr = (fin or {}).get('rooms', {}).get(r['id'])
            if not rr:
                continue
            for j, t in enumerate((str(rr['name']), '%.1f m2' % rr['area'])):
                (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                cv2.putText(out, t, (q(r['cx']) - tw // 2, q(r['cy']) + j * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 230, 207), 1, cv2.LINE_AA)
    return cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].reshape(-1)


async def run(data, area, project, unit, js):
    """data: bytes de la imagen (JPG/PNG). area: m² o None. js: objeto con infer, ocr, progress y preview.
    Devuelve (pdf: Uint8Array, resumen: str JSON)."""
    raw = data.to_py() if hasattr(data, 'to_py') else data
    im0 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if im0 is None:
        raise ValueError('bad_image')
    buf = io.BytesIO()
    def progress(stage, **data):
        if not data:
            js.progress(stage)
            return
        try:
            js.preview(stage, to_js(_preview(**data)))
        except Exception:
            pass                                   # una vista previa nunca debe tumbar el plano

    gen = pipeline.steps(im0, area, project, unit, buf, progress=progress)
    try:
        req = next(gen)
        while True:
            if req[0] == 'infer':
                x = req[2]; n = x.shape[0]
                out = await js.infer(req[1], to_js(np.ascontiguousarray(x).reshape(-1)), n)
                res = np.frombuffer(out.to_py(), np.float32).reshape(n, n, -1)
            elif req[0] == 'ocr':
                pngs = [to_js(cv2.imencode('.png', im)[1].reshape(-1)) for im in req[1]]
                rows = (await js.ocr(to_js(pngs))).to_py()
                res = [[(str(t), float(c), int(x), int(y), int(w), int(h)) for t, c, x, y, w, h in rr] for rr in rows]
            else:
                raise ValueError(req[0])
            req = gen.send(res)
    except StopIteration as e:
        r = e.value
    except ValueError as e:
        if 'area' in str(e).lower():
            raise NoArea(str(e))
        raise
    info = r['info']
    rooms = [dict(name=x['name'], area=round(float(x['area']), 2)) for x in r['fin']['rooms'].values()]
    pdf = buf.getvalue()
    summary = dict(sheets=pdf.count(b'/Type /Page') - pdf.count(b'/Type /Pages'), W=round(float(r['W']), 2), H=round(float(r['H']), 2), area=info.get('area', area),
                   area_read=area is None, rooms=rooms)
    return to_js(pdf), json.dumps(summary)
