"""Browser-equivalent decoder. Stdlib only, so CI can check actual Python/TS parity."""
from pathlib import Path
import json
import math

DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / 'lib/postprocess-config.json').read_text())
CLASSES = ['unreif', 'reif', 'überreif']

def overlap(a, b):
    w = max(0, min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x']))
    h = max(0, min(a['y'] + a['height'], b['y'] + b['height']) - max(a['y'], b['y']))
    return w*h / max(1e-10, a['width']*a['height'] + b['width']*b['height'] - w*h)

def decode(data, size, width, height, captured_at=0, thresholds=None):
    thresholds = {**DEFAULTS, **(thresholds or {})}
    if not all(math.isfinite(n) and n > 0 for n in [size, width, height]):
        return []
    ratio = min(size/width, size/height)
    candidates = []
    for p in range(0, len(data)-7, 8):
        row = data[p:p+8]
        if not all(math.isfinite(x) for x in row) or row[2] <= 0 or row[3] <= 0:
            continue
        ranked = sorted(range(3), key=lambda c: -row[5+c])
        score = row[4]*row[5+ranked[0]]
        if score < thresholds['detection']:
            continue
        left, top = (row[0]-row[2]/2)/ratio, (row[1]-row[3]/2)/ratio
        x, y = max(0, min(width, left)), max(0, min(height, top))
        right, bottom = max(0, min(width, left+row[2]/ratio)), max(0, min(height, top+row[3]/ratio))
        if right-x < 3 or bottom-y < 3:
            continue
        certain = score >= thresholds['classification'] and row[5+ranked[0]]-row[5+ranked[1]] >= thresholds['margin']
        candidates.append({
            'box': {'x': x/width, 'y': y/height, 'width': (right-x)/width, 'height': (bottom-y)/height},
            'predictedClass': ranked[0], 'ripeness': CLASSES[ranked[0]] if certain else 'unsicher',
            'confidence': score, 'capturedAt': captured_at,
            'ripeEvidence': certain and ranked[0] == 1 and score >= max(DEFAULTS['classification'], thresholds['classification']),
            # Metadata retained for the offline evaluator, stripped by CLI parity output.
            '_classMargin': row[5+ranked[0]]-row[5+ranked[1]],
            '_classProbability': row[5+ranked[0]], '_objectness': row[4],
        })
    kept = []
    for candidate in sorted(candidates, key=lambda c: -c['confidence']):
        if not any(overlap(candidate['box'], other['box']) > thresholds['nmsIou'] for other in kept):
            kept.append(candidate)
        if len(kept) == 40:
            break
    return kept

if __name__ == '__main__':
    import sys
    requests = json.load(sys.stdin)
    output = []
    for request in requests:
        result = decode(request['data'], request['size'], request['width'], request['height'], request.get('capturedAt', 0), request.get('thresholds'))
        output.append([{k: v for k, v in d.items() if not k.startswith('_')} for d in result])
    json.dump(output, sys.stdout, ensure_ascii=False)
