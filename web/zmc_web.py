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


async def run(data, area, project, unit, js):
    """data: bytes de la imagen (JPG/PNG). area: m² o None. js: objeto con infer, ocr y progress.
    Devuelve (pdf: Uint8Array, resumen: str JSON)."""
    raw = data.to_py() if hasattr(data, 'to_py') else data
    im0 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if im0 is None:
        raise ValueError('bad_image')
    buf = io.BytesIO()
    gen = pipeline.steps(im0, area, project, unit, buf, progress=lambda s: js.progress(s))
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
