"""
make_dataset.py - Genera le classi 'fractals' e 'kaleidoscope' (sintetiche).
La classe 'eyes' la devi riempire tu con foto vere.

Struttura finale:
  data/
    eyes/         <- TUE immagini di occhi (jpg/png)
    fractals/     <- generate da questo script
    kaleidoscope/ <- generate da questo script

Uso:
  python make_dataset.py --out data --n 2000 --size 128
  (opzionale) python make_dataset.py --out data --n 2000 --kaleido-from data/eyes
"""
import argparse
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ----------------------------------------------------------------------
# Colori: palette coseno casuale (nessuna dipendenza da matplotlib)
# ----------------------------------------------------------------------
def random_palette(rng):
    a = np.array([rng.uniform(0.3, 0.7) for _ in range(3)])
    b = np.array([rng.uniform(0.3, 0.6) for _ in range(3)])
    c = np.array([rng.uniform(0.5, 2.5) for _ in range(3)])
    d = np.array([rng.uniform(0.0, 1.0) for _ in range(3)])

    def pal(t):
        t = t[..., None]
        return np.clip(a + b * np.cos(2 * math.pi * (c * t + d)), 0, 1)

    return pal


# ----------------------------------------------------------------------
# FRATTALI: Julia, Mandelbrot (zoom casuale), Burning Ship
# ----------------------------------------------------------------------
def fractal_image(size, rng):
    kind = rng.choice(["julia", "mandelbrot", "ship"])
    max_iter = rng.randint(60, 140)

    if kind == "julia":
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0.70, 0.80)
        c = complex(rad * math.cos(ang), rad * math.sin(ang))
        span = rng.uniform(1.0, 3.0)
        cx, cy = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
    elif kind == "mandelbrot":
        # punti vicino al bordo dell'insieme, zoom casuale
        cx, cy = rng.choice([(-0.75, 0.1), (-0.16, 1.04), (-1.25, 0.02),
                             (0.28, 0.01), (-0.74, 0.18), (-1.77, 0.0)])
        cx += rng.uniform(-0.03, 0.03)
        cy += rng.uniform(-0.03, 0.03)
        span = 10 ** rng.uniform(-2.2, 0.3)
        c = None
    else:
        cx, cy = rng.choice([(-1.76, -0.03), (-1.62, -0.03), (-1.86, -0.0)])
        cx += rng.uniform(-0.02, 0.02)
        cy += rng.uniform(-0.02, 0.02)
        span = 10 ** rng.uniform(-1.8, 0.0)
        c = None

    xs = np.linspace(cx - span / 2, cx + span / 2, size)
    ys = np.linspace(cy - span / 2, cy + span / 2, size)
    X, Y = np.meshgrid(xs, ys)
    Z0 = X + 1j * Y

    if kind == "julia":
        Z, C = Z0.copy(), np.full_like(Z0, c)
    else:
        Z, C = np.zeros_like(Z0), Z0

    count = np.zeros(Z.shape, dtype=np.float64)
    alive = np.ones(Z.shape, dtype=bool)
    for i in range(max_iter):
        if kind == "ship":
            Z = np.where(alive, (np.abs(Z.real) + 1j * np.abs(Z.imag)) ** 2 + C, Z)
        else:
            Z = np.where(alive, Z * Z + C, Z)
        escaped = alive & (np.abs(Z) > 4)
        # smooth coloring
        with np.errstate(divide="ignore", invalid="ignore"):
            smooth = i + 1 - np.log(np.log(np.abs(Z) + 1e-9) + 1e-9) / math.log(2)
        count[escaped] = smooth[escaped]
        alive &= ~escaped
        Z[~alive & ~escaped] = 0  # evita overflow

    t = count / max_iter
    t = (t * rng.uniform(1.0, 6.0) + rng.random()) % 1.0
    img = random_palette(rng)(t)
    img[alive] = rng.choice([0.0, 0.0, 0.05])  # interno dell'insieme scuro
    arr = (img * 255).astype(np.uint8)

    im = Image.fromarray(arr)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im.rotate(rng.choice([0, 90, 180, 270]))


# ----------------------------------------------------------------------
# CALEIDOSCOPIO: sorgente casuale + piegatura angolare
# ----------------------------------------------------------------------
def random_source(rng, s=384):
    """Immagine sorgente fatta di forme colorate sfocate."""
    bg = tuple(rng.randint(0, 255) for _ in range(3))
    im = Image.new("RGB", (s, s), bg)
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(10, 40)):
        col = tuple(rng.randint(0, 255) for _ in range(3))
        x, y = rng.randint(0, s), rng.randint(0, s)
        w, h = rng.randint(10, s // 3), rng.randint(10, s // 3)
        shape = rng.choice(["ellipse", "rect", "poly", "line"])
        if shape == "ellipse":
            d.ellipse([x, y, x + w, y + h], fill=col)
        elif shape == "rect":
            d.rectangle([x, y, x + w, y + h], fill=col)
        elif shape == "poly":
            pts = [(x + rng.randint(-w, w), y + rng.randint(-h, h)) for _ in range(4)]
            d.polygon(pts, fill=col)
        else:
            d.line([x, y, x + rng.randint(-w, w), y + rng.randint(-h, h)],
                   fill=col, width=rng.randint(2, 12))
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 2)))


def kaleidoscope(src, size, rng):
    src_arr = np.asarray(src.convert("RGB"))
    sh, sw = src_arr.shape[:2]
    n = rng.randint(4, 14)                     # numero di spicchi
    rot = rng.uniform(0, 2 * math.pi)
    zoom = rng.uniform(0.35, 0.9)
    ox, oy = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    x = (xs - size / 2) / (size / 2)
    y = (ys - size / 2) / (size / 2)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x) + rot
    wedge = 2 * math.pi / n
    theta = np.abs(((theta % wedge) + wedge) % wedge - wedge / 2)

    u = (r * np.cos(theta) * zoom + 0.5 + ox) % 1.0
    v = (r * np.sin(theta) * zoom + 0.5 + oy) % 1.0
    out = src_arr[(v * (sh - 1)).astype(int), (u * (sw - 1)).astype(int)]
    return Image.fromarray(out)


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=2000, help="immagini per classe")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kaleido-from", default=None,
                   help="cartella di immagini da usare come sorgente (opzionale)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    fdir = os.path.join(args.out, "fractals")
    kdir = os.path.join(args.out, "kaleidoscope")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(kdir, exist_ok=True)
    os.makedirs(os.path.join(args.out, "eyes"), exist_ok=True)

    extra = []
    if args.kaleido_from:
        extra = [os.path.join(args.kaleido_from, f)
                 for f in os.listdir(args.kaleido_from)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for i in range(args.n):
        fractal_image(args.size, rng).save(os.path.join(fdir, f"f_{i:05d}.png"))

        if extra and rng.random() < 0.4:
            src = Image.open(rng.choice(extra)).convert("RGB")
        else:
            src = random_source(rng)
        kaleidoscope(src, args.size, rng).save(os.path.join(kdir, f"k_{i:05d}.png"))

        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{args.n}")

    print(f"Fatto. Ora metti le foto di occhi in {os.path.join(args.out, 'eyes')}")


if __name__ == "__main__":
    main()
