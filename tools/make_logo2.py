"""IndicPhotoOCR 2.0 wordmark: the original IndicPhotoOCR logo (static/pics/IndicPhotoOCR_LOGO.png of the
IndicPhotoOCR repo) with "2.0" added in Kalam Light (SIL OFL 1.1), the same handwriting family as the original.
Usage: python make_logo2.py <original_logo.png> <Kalam-Light.ttf> <out_dir>"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src_path, font_path, out_dir = sys.argv[1:4]
src = Image.open(src_path).convert("RGBA")
a = np.array(src)[..., 3] > 128
ys, xs = np.where(a)
right, base, top = xs.max(), ys.max(), ys.min()   # wordmark ends at x=942, letters sit on y=157
size = 164                                        # "0" as tall as "OCR"
S, gap = 4, 30

def make(color, out):
    f = ImageFont.truetype(font_path, size)
    bb = f.getbbox("2.0")
    w, h = right + gap + (bb[2] - bb[0]) + 40, src.height
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    im.alpha_composite(src.resize((src.width * S, src.height * S), Image.LANCZOS), (0, 0))
    fs = ImageFont.truetype(font_path, size * S)
    b = fs.getbbox("2.0")
    ImageDraw.Draw(im).text(((right + gap) * S - b[0], base * S - b[3]), "2.0", font=fs, fill=color,
                            stroke_width=1, stroke_fill=color)
    im = im.resize((w, h), Image.LANCZOS)
    al = np.array(im)[..., 3]
    yy, xx = np.where(al > 8)
    im = im.crop((max(0, xx.min() - 16), max(0, yy.min() - 16), min(w, xx.max() + 17), min(h, yy.max() + 17)))
    im.save(os.path.join(out_dir, out))
    print(out, im.size)

make("#23266E", "logo_indicphotoocr2.png")        # light backgrounds (app navy)
make("#FFFFFF", "logo_indicphotoocr2_dark.png")   # dark backgrounds (GitHub dark mode)
