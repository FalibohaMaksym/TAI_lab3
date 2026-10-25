import os, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import ImageFont, Image, ImageDraw

STUDENT = "Faliboha Maksym"
STUDENT_ID = 981
VARIANT = 15
SEED = 1000 * VARIANT + STUDENT_ID

FACTOR = "saltpepper"
LEVELS = [0, 1, 2, 5, 10, 15]

rng = np.random.default_rng(SEED)

FONTS = [  # шрифти з підтримкою кирилиці (Windows)
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

with open("corpus.txt", encoding="utf-8") as f:
    corpus = [s.strip() for s in f if 20 <= len(s.strip()) <= 70]
lines = [str(s) for s in rng.choice(corpus, size=40, replace=False)]

def render_line(text, font_path, size=32, pad=12):
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=0)
    return np.array(img)

def degrade(img, kind, level, rng):
    out = img.astype(np.float32)
    if kind == "noise" and level > 0:           # level: σ гаусового шуму
        out = out + rng.normal(0, level, out.shape)
    elif kind == "blur" and level > 0:          # level: σ гаусового розмиття, пікселі
        out = cv2.GaussianBlur(out, (0, 0), level)
    elif kind == "rotate" and level != 0:       # level: кут повороту, градуси
        h, w = out.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), level, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        out = cv2.warpAffine(out, M, (nw, nh), borderValue=255)
    elif kind == "scale" and level > 1:         # level: у скільки разів зменшити
        h, w = out.shape
        small = cv2.resize(out, (int(w / level), int(h / level)),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "jpeg" and level < 100:        # level: якість JPEG, %
        ok, enc = cv2.imencode(".jpg", out.astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    elif kind == "saltpepper" and level > 0:    # level: % пошкоджених пікселів
        mask = rng.random(out.shape)            # випадкове число 0..1 для кожного пікселя
        out[mask < level / 200] = 0             # "перець": половина пошкоджених пікселів
        out[mask > 1 - level / 200] = 255       # "сіль": друга половина
    return np.clip(out, 0, 255).astype(np.uint8)
records = []
for i, text in enumerate(lines):
    font = FONTS[i % len(FONTS)]
    clean = render_line(text, font, size=int(rng.integers(22, 36)))
    for lvl in LEVELS:
        path = f"data/{i:03d}_{FACTOR}_{lvl}.png"
        cv2.imwrite(path, degrade(clean, FACTOR, lvl, rng))
        records.append({"id": i, "level": lvl, "path": path, "gt": text,
                        "font": os.path.basename(font)})
ds = pd.DataFrame(records)
ds.to_csv("data/ground_truth.csv", index=False)
with open("data/ground_truth.csv", "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()[:16]
print("Записів:", len(ds), "| SHA-256 еталону:", sha)
fig, axes = plt.subplots(len(LEVELS), 1, figsize=(7, 0.75 * len(LEVELS)))
for ax, lvl in zip(axes, LEVELS):
    img = cv2.imread(f"data/000_{FACTOR}_{lvl}.png", cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{lvl} %", fontsize=8, loc="left")
    ax.axis("off")
fig.savefig("figures/fig0_samples.png", dpi=150, bbox_inches="tight")
