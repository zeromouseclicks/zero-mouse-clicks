"""Resumen del banco de medidas: python bench/measure/summary.py resultados.json [más.json ...]"""
import sys
import json
import numpy as np


def stats(v):
    v = [abs(x) for x in v if x is not None]
    if not v:
        return '—'
    return f'{np.median(v):.1f} / {np.percentile(v, 90):.1f}  (n={len(v)})'


def summarize(R, cfg, variants=None):
    rows = []
    for k, r in R.items():
        if variants and r['variant'] not in variants:
            continue
        rows.append(r)
    ok = [r for r in rows if cfg in r['configs'] and 'error' not in r['configs'][cfg]]
    ocr_tot = sum(len(r['ocr']) for r in rows)
    ocr_ok = sum(bool(v) for r in rows for v in r['ocr'].values())
    span_p, span_n, loo, area = [], [], [], []
    found = merged = total = 0
    adj_n = []
    for r in ok:
        c = r['configs'][cfg]
        if c.get('adjust'):
            adj_n.append(c['adjust'].get('n', 0))
        for row in c['rows']:
            total += 1
            found += bool(row.get('found'))
            merged += bool(row.get('merged'))
            if 'ex' in row:
                tgt = span_p if row['printed'] else span_n
                tgt += [100 * row['ex'], 100 * row['ey']]
                area.append(100 * row['earea'])
        for i, l in c.get('loo', {}).items():
            if 'ex' in l:
                loo += [100 * l['ex'], 100 * l['ey']]
    return dict(plans=len(rows), scaled=len(ok), ocr=f'{ocr_ok}/{ocr_tot}', rooms=total, found=found, merged=merged,
                span_printed=stats(span_p), span_other=stats(span_n), loo=stats(loo), area=stats(area),
                adjusted=f'{sum(1 for a in adj_n if a)} plans' if adj_n else '—')


def main():
    R = {}
    for p in sys.argv[1:]:
        R.update(json.load(open(p)))
    cfgs = sorted({c for r in R.values() for c in r['configs']})
    variants = sorted({r['variant'] for r in R.values()})
    print('| variant | config | plans | scaled | OCR dims ok | rooms (merged) | span err printed rooms, cm (median / p90) | span err other rooms | leave-one-out | room area err % |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for v in variants + ['ALL']:
        for cfg in cfgs:
            s = summarize(R, cfg, None if v == 'ALL' else [v])
            print(f"| {v} | {cfg} | {s['plans']} | {s['scaled']} | {s['ocr']} | {s['rooms']} ({s['merged']}) | {s['span_printed']} | {s['span_other']} | {s['loo']} | {s['area']} |")


if __name__ == '__main__':
    main()
