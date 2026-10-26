# %% імпорти та налаштування
import os, glob, time, json, platform
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import cv2
import numpy as np
import pandas as pd
import easyocr
import paddleocr
import paddle
from paddleocr import PaddleOCR
from PIL import Image, ExifTags

from common import (STUDENT, VARIANT, SEED, FACTOR, LEVELS,
                    CONFIGS, CONFIGS_EXTRA, EXTRA_LEVELS, preprocess)

OUT_DIR = os.environ.get("OCR_OUT_DIR", "results")
LIMIT = int(os.environ.get("OCR_LIMIT", "0"))      # 0 = усі зразки; >0 лише для швидкої перевірки
os.makedirs(OUT_DIR, exist_ok=True)

# %% рушії розпізнавання
# EasyOCR: uk+en для синтетики і власних фото, лише en для SROIE (англійські чеки)
reader_uk = easyocr.Reader(["uk", "en"], gpu=False, verbose=False)
reader_en = easyocr.Reader(["en"], gpu=False, verbose=False)

# PaddleOCR 3.x: lang="uk" вибирає eslav-модель; enable_mkldnn=False обходить помилку oneDNN на CPU
PADDLE_ARGS = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                   use_textline_orientation=False, enable_mkldnn=False)
paddle_uk = PaddleOCR(lang="uk", **PADDLE_ARGS)
paddle_en = PaddleOCR(lang="en", **PADDLE_ARGS)
# для великих фото (довга сторона 2000 px) обмежуємо довгу сторону детектора: втричі швидше на CPU
PADDLE_PAGE_ARGS = dict(PADDLE_ARGS, text_det_limit_type="max", text_det_limit_side_len=1280)
paddle_uk_page = PaddleOCR(lang="uk", **PADDLE_PAGE_ARGS)


def ocr_easyocr(reader, img, page=False):
    t0 = time.perf_counter()
    res = reader.readtext(img, detail=1)
    if page:    # сторінка: фрагменти за рядками (допуск 25 пікселів), потім зліва направо
        res = sorted(res, key=lambda r: (round(r[0][0][1] / 25), r[0][0][0]))
    else:       # один рядок: зліва направо
        res = sorted(res, key=lambda r: min(p[0] for p in r[0]))
    text = " ".join(r[1] for r in res)
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return text, conf, time.perf_counter() - t0


def ocr_paddle(engine, img, page=False):
    t0 = time.perf_counter()
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img   # Paddle приймає лише 3 канали
    r = engine.predict(bgr)[0]
    items = list(zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]))
    if page:
        items.sort(key=lambda it: (round(float(np.min(it[0][:, 1])) / 25), float(np.min(it[0][:, 0]))))
    else:
        items.sort(key=lambda it: float(np.min(it[0][:, 0])))
    text = " ".join(it[1] for it in items)
    conf = float(np.mean([it[2] for it in items])) if items else 0.0
    return text, conf, time.perf_counter() - t0


ENGINES = {
    "synthetic": {"EasyOCR": lambda im: ocr_easyocr(reader_uk, im),
                  "PaddleOCR": lambda im: ocr_paddle(paddle_uk, im)},
    "sroie":     {"EasyOCR": lambda im: ocr_easyocr(reader_en, im),
                  "PaddleOCR": lambda im: ocr_paddle(paddle_en, im)},
    "own":       {"EasyOCR": lambda im: ocr_easyocr(reader_uk, im, page=True),
                  "PaddleOCR": lambda im: ocr_paddle(paddle_uk_page, im, page=True)},
}


def run_group(group, samples, load, configs_for):
    """samples: список словників з ключами id, level, gt, ...; load(sample) -> зображення сірого."""
    rows = []
    for n, s in enumerate(samples, 1):
        img = load(s)
        for cfg, steps in configs_for(s).items():
            prepared = preprocess(img, steps)
            for eng, fn in ENGINES[group].items():
                try:
                    text, conf, sec = fn(prepared)
                    err = ""
                except Exception as e:      # збій одного зразка не повинен втрачати весь експеримент
                    text, conf, sec, err = "", 0.0, 0.0, repr(e)[:200]
                rows.append({**s, "group": group, "config": cfg, "engine": eng,
                             "pred": text, "conf": conf, "sec": sec, "error": err})
        if n % 10 == 0 or n == len(samples):
            print(f"[{group}] {n}/{len(samples)}", flush=True)
    return pd.DataFrame(rows)


# %% синтетичний набір (з data/ground_truth.csv)
gt = pd.read_csv("data/ground_truth.csv")
if LIMIT:
    gt = gt[gt["id"] < LIMIT]
synth = [{"id": r.id, "level": r.level, "path": r.path, "gt": r.gt} for r in gt.itertuples()]


def synth_configs(s):
    cfgs = dict(CONFIGS)
    if s["level"] in EXTRA_LEVELS:
        cfgs.update(CONFIGS_EXTRA)
    return cfgs


res_synth = run_group("synthetic", synth,
                      lambda s: cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE), synth_configs)
res_synth.to_csv(f"{OUT_DIR}/results_synthetic.csv", index=False, encoding="utf-8")

# %% відкритий набір SROIE (з data/sroie_ids.csv)
sroie = pd.read_csv("data/sroie_ids.csv")
if LIMIT:
    sroie = sroie.head(LIMIT)
sroie_samples = [{"id": r.idx, "level": 0, "path": r.path, "gt": r.gt} for r in sroie.itertuples()]
res_sroie = run_group("sroie", sroie_samples,
                      lambda s: cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE), lambda s: CONFIGS)
res_sroie.to_csv(f"{OUT_DIR}/results_sroie.csv", index=False, encoding="utf-8")


# %% власні фотографії (data/own_photos/*.jpg + однойменні .txt)
def exif_info(path):
    """Дата зйомки і модель камери з EXIF (доказ, що фото зроблено студентом)."""
    exif = Image.open(path).getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    taken = exif.get_ifd(0x8769).get(0x9003) or tags.get("DateTime")
    return taken, tags.get("Model")


def load_own(s):
    gray = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE)
    k = 2000 / max(gray.shape)                      # довга сторона 2000 пікселів
    return cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


own = []
for i, p in enumerate(sorted(glob.glob("data/own_photos/*.jpg"))):
    with open(p[:-4] + ".txt", encoding="utf-8") as f:
        gt_text = f.read()
    taken, camera = exif_info(p)
    own.append({"id": i, "level": 0, "path": p, "file": os.path.basename(p), "gt": gt_text,
                "kind": "page" if os.path.basename(p).startswith("page") else "label",
                "taken": taken, "camera": camera})
if LIMIT:
    own = own[:LIMIT]
res_own = run_group("own", own, load_own, lambda s: CONFIGS)
res_own.to_csv(f"{OUT_DIR}/results_own.csv", index=False, encoding="utf-8")

# %% метадані запуску (версії, параметри)
meta = {
    "student": STUDENT, "variant": VARIANT, "seed": SEED, "factor": FACTOR, "levels": LEVELS,
    "python": platform.python_version(),
    "easyocr": easyocr.__version__, "paddleocr": paddleocr.__version__, "paddlepaddle": paddle.__version__,
    "opencv": cv2.__version__, "numpy": np.__version__,
    "easyocr_langs": {"synthetic/own": ["uk", "en"], "sroie": ["en"]},
    "paddle_lang": {"synthetic/own": "uk", "sroie": "en"},
    "paddle_args": PADDLE_ARGS,
    "paddle_page_args": PADDLE_PAGE_ARGS,
    "configs": {k: list(v) for k, v in {**CONFIGS, **CONFIGS_EXTRA}.items()},
    "nlmeans_params": {"h": 20, "templateWindowSize": 7, "searchWindowSize": 21},
    "extra_configs_levels": EXTRA_LEVELS,
    "limit": LIMIT,
}
with open(f"{OUT_DIR}/run_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("Готово. Результати в", OUT_DIR)
