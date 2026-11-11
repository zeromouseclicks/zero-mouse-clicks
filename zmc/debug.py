"""Imagen de depuración: original | máscara del modelo | modelo vectorial con nombres y áreas."""
import cv2, numpy as np
PAL = np.array([[255, 255, 255], [200, 230, 255], [30, 30, 30], [0, 140, 255], [255, 80, 40],
                [40, 160, 40], [200, 60, 200], [150, 210, 160], [120, 200, 120]], np.uint8)
def render(im, m, mod=None, fin=None, f=2):
    a = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(PAL[m], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    c = np.full_like(a, 255)
    if mod is not None:
        for o, x0, y0, x1, y1 in mod['walls']:
            cv2.rectangle(c, (x0*f, y0*f), (x1*f-1, y1*f-1), (60, 60, 60), -1)
            cv2.rectangle(b, (x0*f, y0*f), (x1*f-1, y1*f-1), (0, 0, 255), 1)
        for op in mod['openings']:
            col = (0, 140, 255) if op['kind'] == 'door' else (255, 120, 40)
            cv2.rectangle(c, (op['x0']*f, op['y0']*f), (op['x1']*f-1, op['y1']*f-1), col, -1)
        for r in mod['rooms']:
            s = str(r['id'])
            if fin: 
                rr = fin['rooms'][r['id']]; s = f"{rr['name']} {rr['area']:.1f}"
            cv2.putText(c, s, (int(r['cx']*f)-40, int(r['cy']*f)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
    return np.hstack([a, b, c])
