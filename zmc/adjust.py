"""Ajuste local por cotas impresas (v0.4, solo reglas, sin tokens).

La escala global (calib.fit_dims) deja cada espacio con el error de la máscara: ~2 px por cara de muro, unos 4 cm.
Aquí las medidas impresas que cuadraron con la escala mandan también en el sitio: se mueven las líneas de muro
lo justo para que cada luz impresa quede igual en el dibujo.

Por eje (X para muros verticales, Y para horizontales):
- incógnitas: el corrimiento Δk de cada línea de muro (caras izquierda y derecha se mueven juntas: el grosor
  no cambia); entre dos líneas el corrimiento se interpola linealmente (pisos, muebles y textos se estiran poco);
- observaciones: cada medida impresa usada en la escala, f(cara alta) − f(cara baja) = valor / escala, con peso
  alto; y la distancia entre líneas vecinas como la leyó el modelo (±1,5 px), con peso bajo;
- mínimos cuadrados; se descartan medidas que piden mover un muro más de lo creíble (otra cota las contradice o
  el folleto no está a escala) y se informan.

Después la imagen y la máscara se remuestrean con ese mapa (separable, monótono) y se vuelve a vectorizar: todas
las capas (muros, vanos, espacios, muebles) quedan coherentes. Con `up` = 2 se trabaja a doble resolución para
que el redondeo al píxel no se coma la mejora.
"""
import numpy as np
import cv2

SIG_GAP = 1.5            # px: incertidumbre de la distancia entre dos líneas de muro vecinas leída por el modelo
SIG_ABS = 6.0            # px: posición absoluta (solo fija el conjunto)
SIG_DIM = 0.15           # px: una medida impresa (al cm) manda casi del todo
MAX_SHIFT = 0.06         # una medida que pide cambiar su luz más de 6 % (o > 6 px) no se usa
MAX_SHIFT_PX = 6.0


def _ray(obs, x, y, dx, dy, lim):
    H, W = obs.shape
    for i in range(1, lim):
        xx, yy = x + dx * i, y + dy * i
        if xx < 0 or yy < 0 or xx >= W or yy >= H:
            return None
        if obs[yy, xx]:
            return i
    return None


def bounds(model, obsA, room, x, y, axis, L):
    """Caras de muro (coordenadas de borde, px) que delimitan la luz de largo ~L medida para `room` cerca de
    (x, y) en el eje 'h' (izq.-der.) o 'v' (arriba-abajo). Solo de muro a muro: una luz que termina en un clóset
    o un mesón no sirve para mover muros. None si no se encuentra o los rayos no coinciden entre sí."""
    bx, by, bw, bh = room['bbox']
    lim = int(max(model['shape']))
    x0, y0 = int(round(x)), int(round(y))
    rays = []
    for f in (-0.25, 0.0, 0.25):
        if axis == 'h':
            yy = int(round(y + f * bh))
            if not (by <= yy < by + bh):
                continue
            l, r = _ray(obsA, x0, yy, -1, 0, lim), _ray(obsA, x0, yy, 1, 0, lim)
            if l and r:
                rays.append((float(x0 - l + 1), float(x0 + r)))
        else:
            xx = int(round(x + f * bw))
            if not (bx <= xx < bx + bw):
                continue
            u, d = _ray(obsA, xx, y0, 0, -1, lim), _ray(obsA, xx, y0, 0, 1, lim)
            if u and d:
                rays.append((float(y0 - u + 1), float(y0 + d)))
    if not rays:
        return None
    lo = float(np.median([q[0] for q in rays])); hi = float(np.median([q[1] for q in rays]))
    if max(abs(q[0] - lo) + abs(q[1] - hi) for q in rays) > 2.0:
        return None                                   # muro con quiebres: la luz no es única
    tol = max(2.5, 0.6 * model['thickness_px'])
    return (lo, hi) if abs((hi - lo) - L) <= tol else None


def lines(model, axis):
    """Líneas de muro perpendiculares al eje: [(cara baja, cara alta)] ordenadas y sin traslapes."""
    o = 'v' if axis == 'h' else 'h'
    iv = []
    for w in model['walls']:
        if w[0] != o:
            continue
        lo, hi = (w[1], w[3]) if axis == 'h' else (w[2], w[4])
        iv.append([float(lo), float(hi)])
    for op in model['openings']:
        if op['orient'] != o:
            continue
        lo, hi = (op['x0'], op['x1']) if axis == 'h' else (op['y0'], op['y1'])
        iv.append([float(lo), float(hi)])
    iv.sort()
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1] + 0.5:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _phi(p, ln):
    """Pesos de interpolación del corrimiento en la posición p sobre las líneas ln -> {k: peso}."""
    if p <= ln[0][1]:
        return {0: 1.0}
    if p >= ln[-1][0]:
        return {len(ln) - 1: 1.0}
    for k in range(len(ln)):
        lo, hi = ln[k]
        if lo <= p <= hi:
            return {k: 1.0}
        if k + 1 < len(ln) and hi < p < ln[k + 1][0]:
            t = (p - hi) / (ln[k + 1][0] - hi)
            return {k: 1.0 - t, k + 1: t}
    return {len(ln) - 1: 1.0}


def solve(ln, obs):
    """obs: [(p_lo, p_hi, target_px)] -> (Δ por línea, usadas, descartadas).
    Lo que leyó el modelo se respeta como distancias entre líneas vecinas (no como posiciones): corregir la luz
    de un espacio corre los muros que siguen en vez de apretar al vecino que no trae medida."""
    K = len(ln)
    used, dropped = [], []
    for p_lo, p_hi, T in obs:
        need = T - (p_hi - p_lo)
        if abs(need) > max(MAX_SHIFT_PX, MAX_SHIFT * T):
            dropped.append((p_lo, p_hi, T, need)); continue
        used.append((p_lo, p_hi, T))
    if not K:
        return np.zeros(0), [], dropped + [o + (0.0,) for o in used]
    for _ in range(4):
        A = np.eye(K) / SIG_ABS ** 2
        for k in range(K - 1):                       # distancias entre líneas vecinas, como las leyó el modelo
            g = np.zeros(K); g[k + 1], g[k] = 1, -1
            A += np.outer(g, g) / SIG_GAP ** 2
        b = np.zeros(K)
        for p_lo, p_hi, T in used:
            g = np.zeros(K)
            for k, w in _phi(p_hi, ln).items():
                g[k] += w
            for k, w in _phi(p_lo, ln).items():
                g[k] -= w
            r = T - (p_hi - p_lo)
            A += np.outer(g, g) / SIG_DIM ** 2
            b += g * r / SIG_DIM ** 2
        d = np.linalg.solve(A, b)
        cross = [k for k in range(K - 1) if (ln[k + 1][0] + d[k + 1]) - (ln[k][1] + d[k]) < 1.0]
        res = [abs((T - (p_hi - p_lo)) - sum(w * d[k] for k, w in _phi(p_hi, ln).items())
                   + sum(w * d[k] for k, w in _phi(p_lo, ln).items())) for p_lo, p_hi, T in used]
        if not cross and (not used or max(res) <= 1.0):
            return d, used, dropped
        # dos medidas que se contradicen (o un cruce de muros): se suelta la que peor queda
        i = int(np.argmax(res)) if used else None
        if i is None:
            return np.zeros(K), [], dropped
        worst = used.pop(i)
        dropped.append(worst + (worst[2] - (worst[1] - worst[0]),))
    return np.zeros(K), [], dropped + [o + (0.0,) for o in used]


class Warp:
    """Mapa monótono por eje. Cada línea de muro se corre un número entero de píxeles (sus dos caras juntas: el
    grosor no cambia y no hace falta volver a vectorizar); entre líneas el corrimiento se interpola."""

    def __init__(self, ln, d, size):
        D = np.round(np.asarray(d, float)).astype(int) if len(ln) else np.zeros(0, int)
        self.ln, self.D = ln, D
        if len(ln):
            self.xo = np.array([v for lo, hi in ln for v in (lo, hi)], float)
            self.dd = np.array([D[k] for k in range(len(ln)) for _ in (0, 1)], float)
        else:
            self.xo, self.dd = np.array([0.0, float(size)]), np.zeros(2)
        self.off = int(max(0, -self.dd.min()))
        self.n_new = int(size + max(0, int(self.dd.max())) + self.off)
        e = 1e5
        self.gx = np.concatenate([[-e], self.xo, [size + e]])
        self.gy = np.concatenate([[-e + self.dd[0]], self.xo + self.dd, [size + e + self.dd[-1]]])

    def fwd(self, x):
        """Coordenada (borde o punto) vieja -> nueva, en px (flotante)."""
        return np.asarray(x, float) + np.interp(x, self.xo, self.dd) + self.off

    def edge(self, x):
        """Coordenada de borde entera -> entera (las caras de muro caen exactas)."""
        return int(round(float(self.fwd(x))))

    def src(self):
        """Píxel de origen (centro) de cada píxel de salida, para cv2.remap."""
        y = np.arange(self.n_new) + 0.5 - self.off
        return (np.interp(y, self.gy, self.gx) - 0.5).astype(np.float32)


def plan(model, mask, fit, like=None):
    """Corrimientos por eje a partir de las medidas usadas en la escala. -> (wx, wy, info) o None."""
    from .calib import obstacles
    if not fit or not fit.get('used'):
        return None
    s = model['m_per_px']
    obsA, obsB = obstacles(model, mask)
    rooms = {r['id']: r for r in model['rooms']}
    obs = {'h': [], 'v': []}
    tags = {'h': [], 'v': []}
    for u in fit['used']:
        r = rooms.get(u['room'])
        if r is None or r.get('exterior') or 'x' not in u:
            continue
        L = u['px']                                    # largo medido (px de la máscara actual)
        bd = bounds(model, obsA, r, u['x'], u['y'], u['axis'], L)
        if bd is None:
            continue
        obs[u['axis']].append((bd[0], bd[1], u['value'] / s))
        tags[u['axis']].append(dict(room=u['room'], value=u['value']))
    if sum(len(v) for v in obs.values()) < 2:
        return None
    H, W = model['shape']
    out = {}
    info = dict(n=0, dropped=[], shift_px=0.0)
    for axis, size in (('h', W), ('v', H)):
        ln = lines(model, axis)
        d, used, dropped = solve(ln, obs[axis])
        out[axis] = (ln, d)
        info['n'] += len(used)
        info['shift_px'] = max(info['shift_px'], float(np.abs(d).max()) if len(d) else 0.0)
        for o in dropped:
            info['dropped'].append(dict(axis=axis, value_m=round(o[2] * s, 3), drawn_m=round((o[1] - o[0]) * s, 3)))
    return out, info


def apply(model, im, mask, fit, like=None):
    """Corre los muros en el modelo vectorial y remuestrea los rásters con el mismo mapa.
    -> (model, im, mask, like, wx, wy, info), o None si no hay nada que ajustar."""
    p = plan(model, mask, fit, like)
    if p is None:
        return None
    out, info = p
    if info['n'] < 2:
        return None
    H, W = mask.shape
    wx = Warp(*out['h'], W)
    wy = Warp(*out['v'], H)
    if not (np.any(wx.D) or np.any(wy.D)):
        return None
    mx = np.broadcast_to(wx.src()[None, :], (wy.n_new, wx.n_new)).astype(np.float32)
    my = np.broadcast_to(wy.src()[:, None], (wy.n_new, wx.n_new)).astype(np.float32)
    rm = lambda a, f=cv2.INTER_NEAREST: cv2.remap(a, mx, my, f, borderMode=cv2.BORDER_REPLICATE)
    im2 = rm(im, cv2.INTER_LINEAR)
    m2 = rm(mask)
    lk2 = rm(like.astype(np.uint8)) > 0 if like is not None else None
    md = dict(model)
    md['walls'] = [[o, wx.edge(a), wy.edge(b), wx.edge(c), wy.edge(d)] for o, a, b, c, d in model['walls']]
    ops = []
    for op in model['openings']:
        q = dict(op)
        q.update(x0=wx.edge(op['x0']), x1=wx.edge(op['x1']), y0=wy.edge(op['y0']), y1=wy.edge(op['y1']))
        ops.append(q)
    md['openings'] = ops
    lab = rm(model['room_labels'].astype(np.int32 if model['room_labels'].dtype != np.uint8 else np.uint8).astype(np.float32)).astype(model['room_labels'].dtype)
    md['room_labels'] = lab
    md['footprint'] = rm(model['footprint'].astype(np.uint8)) > 0
    rooms = []
    for r in model['rooms']:
        q = dict(r)
        ys, xs = np.nonzero(lab == r['label'])
        if xs.size:
            q.update(area_px=int(xs.size), cx=float(xs.mean()), cy=float(ys.mean()),
                     bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])
        if r.get('texts'):
            q['texts'] = [dict(t, mx=float(wx.fwd(t['mx'])), my=float(wy.fwd(t['my']))) for t in r['texts']]
        rooms.append(q)
    md['rooms'] = rooms
    md['shape'] = [wy.n_new, wx.n_new]
    ws = md['walls']
    if ws:
        md['bbox'] = [min(w[1] for w in ws), min(w[2] for w in ws), max(w[3] for w in ws), max(w[4] for w in ws)]
    md['inner_px'] = sum(r['area_px'] for r in rooms if not r.get('exterior')) or int(md['footprint'].sum())
    md.pop('net_px', None)
    info['shift_px'] = max(int(np.abs(wx.D).max()) if len(wx.D) else 0, int(np.abs(wy.D).max()) if len(wy.D) else 0)
    return md, im2, m2, lk2, wx, wy, info


def check(model, mask, fit, wx, wy):
    """Cuánto se separa el dibujo final de cada medida impresa usada (cm): [mediana, máximo]."""
    from .calib import obstacles, spans
    from .texts import _room_at
    s = model['m_per_px']
    obsA, obsB = obstacles(model, mask)
    by_label = {r['label']: r for r in model['rooms']}
    err = []
    for u in fit['used']:
        x, y = float(wx.fwd(u['x'])), float(wy.fwd(u['y']))
        r = by_label.get(_room_at(model['room_labels'], x, y, int(2.5 * model['thickness_px'])))
        if r is None or r.get('exterior'):
            continue
        sp = spans(model, obsA, obsB, r, x, y)[u['axis']]
        if sp:
            err.append(min(abs(L * s - u['value']) for L in sp) * 100)
    return [round(float(np.median(err)), 2), round(float(np.max(err)), 2)] if err else None
