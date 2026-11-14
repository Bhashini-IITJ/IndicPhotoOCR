"""Light/dark versions of the IIT Jodhpur and Bhashini logos for the README.
Light = the original transparent PNG, trimmed. Dark = navy/blue parts turned white, orange/green kept,
and (IITJ) the white fills made transparent so the seal sits on the dark page.
Usage: python make_dark_logos.py <logo_iitj.png> <logo_bhashini.png> <out_dir>"""
import sys, os
import numpy as np
from PIL import Image

def trim(im, pad=4):
    a = np.array(im)[..., 3]; ys, xs = np.where(a > 8)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), xs.max() + pad + 1, ys.max() + pad + 1))

def warm(rgb):   # orange or green parts that keep their colour
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    orange = (r > 150) & (r - b > 90)
    green = (g > 90) & (g - r > 50) & (g - b > 20)
    return orange | green

def dark_iitj(im):
    a = np.array(im.convert("RGBA")).astype(float)
    rgb, al = a[..., :3], a[..., 3]
    keep = warm(rgb)
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255
    out = a.copy()
    m = ~keep
    out[m, :3] = 255                         # navy -> white
    out[m, 3] = al[m] * np.clip((1 - lum[m]) / 0.85, 0, 1)   # white fills -> transparent
    return Image.fromarray(out.astype(np.uint8))

def dark_bhashini(im):
    a = np.array(im.convert("RGBA")).astype(float)
    rgb = a[..., :3]
    blue = (rgb[..., 2] > 100) & (rgb[..., 2] - rgb[..., 0] > 60) & ~warm(rgb)
    out = a.copy(); out[blue, :3] = 255
    return Image.fromarray(out.astype(np.uint8))

iitj, bhashini, out = sys.argv[1:4]
for name, path, fn in [("iitj", iitj, dark_iitj), ("bhashini", bhashini, dark_bhashini)]:
    im = trim(Image.open(path).convert("RGBA"))
    im.save(os.path.join(out, f"logo_{name}.png"))
    trim(fn(im)).save(os.path.join(out, f"logo_{name}_dark.png"))
    print(name, im.size)
