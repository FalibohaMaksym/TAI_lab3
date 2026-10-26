# Додатковий набір до питання 2 (підрозділ 7.3): ті самі умови, що й в основному наборі, але інші рядки,
# лише найгірший рівень шуму і лише конфігурація raw. Основний набір і його хеш не змінюються.
# %% імпорти та налаштування
import os, time, hashlib
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import cv2
import numpy as np
import pandas as pd
import easyocr
from paddleocr import PaddleOCR
from PIL import ImageFont, Image, ImageDraw

from common import SEED, FACTOR

LEVEL = int(os.environ.get("EXTRA_LEVEL", "15"))
N_EXTRA = int(os.environ.get("EXTRA_N", "120"))
EXTRA_SEED = SEED + 1                       # інше насіння, ніж в основному наборі
OUT_DIR = os.environ.get("OCR_OUT_DIR", "results")
DATA_DIR = "data_extra"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]

# %% вибір рядків, яких немає в основному наборі
used = set(pd.read_csv("data/ground_truth.csv")["gt"])
with open("corpus.txt", encoding="utf-8") as f:
    corpus = [s.strip() for s in f if 20 <= len(s.strip()) <= 70 and s.strip() not in used]
rng = np.random.default_rng(EXTRA_SEED)
lines = [str(s) for s in rng.choice(corpus, size=N_EXTRA, replace=False)]


def render_line(text, font_path, size=32, pad=12):       # як у make_synthetic.py
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=0)
    return np.array(img)


def saltpepper(img, level, rng):                          # як гілка saltpepper у degrade
    out = img.astype(np.float32)
    mask = rng.random(out.shape)
    out[mask < level / 200] = 0
    out[mask > 1 - level / 200] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


records = []
for i, text in enumerate(lines):
    font = FONTS[i % len(FONTS)]
    clean = render_line(text, font, size=int(rng.integers(22, 36)))
    path = f"{DATA_DIR}/{i:03d}_{FACTOR}_{LEVEL}.png"
    cv2.imwrite(path, saltpepper(clean, LEVEL, rng))
    records.append({"id": 1000 + i, "level": LEVEL, "path": path, "gt": text, "font": os.path.basename(font)})
ds = pd.DataFrame(records)
ds.to_csv(f"{DATA_DIR}/ground_truth_extra.csv", index=False)
with open(f"{DATA_DIR}/ground_truth_extra.csv", "rb") as f:
    print("Записів:", len(ds), "| SHA-256 додаткового еталону:", hashlib.sha256(f.read()).hexdigest()[:16])

# %% розпізнавання (ті самі інструменти й параметри, що в run_experiment.py)
reader_uk = easyocr.Reader(["uk", "en"], gpu=False, verbose=False)
paddle_uk = PaddleOCR(lang="uk", use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=False, enable_mkldnn=False)


def ocr_easyocr(img):
    t0 = time.perf_counter()
    res = sorted(reader_uk.readtext(img, detail=1), key=lambda r: min(p[0] for p in r[0]))
    text = " ".join(r[1] for r in res)
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return text, conf, time.perf_counter() - t0


def ocr_paddle(img):
    t0 = time.perf_counter()
    r = paddle_uk.predict(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[0]
    items = sorted(zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]), key=lambda it: float(np.min(it[0][:, 0])))
    text = " ".join(it[1] for it in items)
    conf = float(np.mean([it[2] for it in items])) if items else 0.0
    return text, conf, time.perf_counter() - t0


rows = []
for n, s in enumerate(records, 1):
    img = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE)
    for eng, fn in {"EasyOCR": ocr_easyocr, "PaddleOCR": ocr_paddle}.items():
        text, conf, sec = fn(img)
        rows.append({**s, "config": "raw", "engine": eng, "pred": text, "conf": conf, "sec": sec})
    if n % 20 == 0 or n == len(records):
        print(f"[extra] {n}/{len(records)}", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/results_extra_level{LEVEL}.csv", index=False, encoding="utf-8")
print("Готово.")
