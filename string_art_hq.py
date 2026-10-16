"""
High-cycle version of addisonkopplin/string_art (string_art.py).

Same greedy algorithm as the original: start at a nail; from the current nail,
try a straight line to every other allowed nail, pick the line whose pixels are
darkest on average in the (remaining) target image, "use up" that darkness, move
to the chosen nail, repeat. Changes vs. the original are listed in CHANGES below.
"""
import argparse, time, json
import numpy as np
from PIL import Image, ImageOps, ImageFilter

CHANGES = """
- Line pixels for every nail pair are precomputed once (the original re-rasterized every
  candidate line into a fresh full-size array each step, which is far too slow for 1000s of lines).
- After a line is chosen, its pixels are lightened by --strength (e.g. 0.12) instead of being set
  to fully white. The original's "set to white" means each pixel can only be used once, so the
  image runs out of dark pixels after a few hundred lines.
- Optional --skip: don't connect to nails within N positions of the current nail (original: 0),
  which avoids short edge-hugging chords.
- Preprocessing: square face crop, autocontrast + gamma, circular mask (outside circle = white).
- Fixed line budget (--lines) instead of the original's 350-line cap / moving-average stop.
"""

def load_target(path, crop, size, gamma, cutoff):
    im = Image.open(path).convert('L')
    if crop: im = im.crop(tuple(crop))
    im = ImageOps.autocontrast(im, cutoff=cutoff)
    im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64) / 255.0
    a = a ** gamma
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    a[(xx - c) ** 2 + (yy - c) ** 2 > (size / 2) ** 2] = 1.0   # white outside the circle
    return a

def nail_xy(n, size):
    c = (size - 1) / 2; r = size / 2 - 1
    t = 2 * np.pi * np.arange(n) / n
    return c + r * np.cos(t), c - r * np.sin(t)

def line_pixels(x0, y0, x1, y1, size):
    L = int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)))) + 1
    xs = np.rint(np.linspace(x0, x1, L)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, L)).astype(np.int32)
    idx = np.unique(ys * size + xs)
    return idx

def run(args):
    t0 = time.time()
    target = load_target(args.input, args.crop, args.size, args.gamma, args.cutoff)
    dark = (1.0 - target).ravel().copy()
    X, Y = nail_xy(args.nails, args.size)
    n = args.nails
    # precompute per-nail concatenated pixel lists to every other nail
    per = []
    for i in range(n):
        lst, offs, js = [], [0], []
        for j in range(n):
            if j == i: continue
            p = line_pixels(X[i], Y[i], X[j], Y[j], args.size)
            lst.append(p); offs.append(offs[-1] + len(p)); js.append(j)
        per.append((np.concatenate(lst), np.array(offs), np.array(js)))
    t_pre = time.time() - t0

    order = [0]; cur = 0; prev = -1; scores = []
    used = set()
    for k in range(args.lines):
        cat, offs, js = per[cur]
        sums = np.add.reduceat(dark[cat], offs[:-1])
        lens = np.diff(offs)
        score = sums / lens
        ring = np.abs(js - cur); ring = np.minimum(ring, n - ring)
        bad = (ring <= args.skip) | (js == prev)
        score[bad] = -1
        if args.no_repeat:
            for m, j in enumerate(js):
                if (min(cur, j), max(cur, j)) in used: score[m] = -1
        b = int(np.argmax(score))
        if score[b] <= args.min_score:
            print(f'stop at {k}: best remaining line darkness {score[b]:.4f}'); break
        j = int(js[b])
        pix = cat[offs[b]:offs[b + 1]]
        dark[pix] = np.maximum(dark[pix] - args.strength, 0.0)
        used.add((min(cur, j), max(cur, j)))
        scores.append(float(score[b]))
        prev, cur = cur, j
        order.append(cur)
    t_run = time.time() - t0 - t_pre
    order = np.array(order, dtype=int)
    np.save(args.out + '.npy', order)
    with open(args.out + '_nail_order.txt', 'w') as f:
        f.write(f'# {len(order)-1} lines, {n} nails numbered 0..{n-1} counter-clockwise starting at 3 o\'clock\n')
        f.write('\n'.join(map(str, order)) + '\n')
    render(order, X, Y, args)
    meta = dict(vars(args), lines_drawn=len(order) - 1, precompute_s=round(t_pre, 1), greedy_s=round(t_run, 1),
                total_s=round(time.time() - t0, 1), final_score=scores[-1] if scores else None)
    json.dump(meta, open(args.out + '_params.json', 'w'), indent=1)
    print(json.dumps(meta))

def render(order, X, Y, args):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    s = args.size
    pts = np.stack([X[order], Y[order]], 1)
    segs = np.stack([pts[:-1], pts[1:]], 1)
    px = args.render_px; dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1, s); ax.set_ylim(s, -1); ax.set_aspect('equal'); ax.axis('off')
    ax.add_collection(LineCollection(segs, colors='black', linewidths=args.lw, alpha=args.alpha, antialiased=True))
    fig.savefig(args.out + '.png', facecolor='white'); plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p.add_argument('--crop', type=int, nargs=4); p.add_argument('--size', type=int, default=500)
    p.add_argument('--gamma', type=float, default=1.0); p.add_argument('--cutoff', type=float, default=1)
    p.add_argument('--nails', type=int, default=240); p.add_argument('--lines', type=int, default=3000)
    p.add_argument('--strength', type=float, default=0.12); p.add_argument('--skip', type=int, default=15)
    p.add_argument('--min_score', type=float, default=0.0); p.add_argument('--no_repeat', action='store_true')
    p.add_argument('--render_px', type=int, default=1600); p.add_argument('--lw', type=float, default=0.25)
    p.add_argument('--alpha', type=float, default=0.35)
    run(p.parse_args())
