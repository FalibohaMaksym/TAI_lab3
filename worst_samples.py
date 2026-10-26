# Аналіз найгірших зразків (пункт 14 розділу 5): зображення, еталон, результат з підсвіткою розбіжностей
# %% імпорти та налаштування
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rapidfuzz.distance import Levenshtein

from common import STUDENT, VARIANT, SEED
from metrics import normalize

RES_DIR = os.environ.get("OCR_OUT_DIR", "results")
FIG_DIR = os.environ.get("FIG_DIR", "figures")
GROUP = os.environ.get("WORST_GROUP", "synthetic")      # synthetic | sroie | own
CONFIG = "raw"                                          # аналізуємо необроблені зображення, як у 4.7
N_WORST = 5
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {"eq": "none", "sub": "#f4a6a6", "del": "#f4a6a6", "ins": "#f6e27a"}   # рожевий: заміна/пропуск, жовтий: вставка


def align(a, b):
    """Вирівнювання двох рядків за операціями редагування: '·' позначає символ, якого немає."""
    A, B, tags = [], [], []
    i = j = 0
    for op in Levenshtein.editops(a, b):
        while i < op.src_pos and j < op.dest_pos:
            A.append(a[i]); B.append(b[j]); tags.append("eq"); i += 1; j += 1
        if op.tag == "replace":
            A.append(a[i]); B.append(b[j]); tags.append("sub"); i += 1; j += 1
        elif op.tag == "delete":
            A.append(a[i]); B.append("·"); tags.append("del"); i += 1
        else:
            A.append("·"); B.append(b[j]); tags.append("ins"); j += 1
    while i < len(a) and j < len(b):
        A.append(a[i]); B.append(b[j]); tags.append("eq"); i += 1; j += 1
    return A, B, tags


def draw_diff(ax, gt, pred):
    gt_n, pred_n = normalize(gt), normalize(pred)
    A, B, tags = align(gt_n, pred_n)
    ax.set_xlim(0, max(len(A), 40)); ax.set_ylim(0, 2); ax.invert_yaxis(); ax.axis("off")
    for k, (a, b, t) in enumerate(zip(A, B, tags)):
        for row, ch in ((0.5, a), (1.5, b)):
            if t != "eq":
                ax.add_patch(Rectangle((k, row - 0.5), 1, 1, color=COLORS[t], lw=0))
            ax.text(k + 0.5, row, ch, ha="center", va="center", family="DejaVu Sans Mono", fontsize=8)
    ax.text(-0.5, 0.5, "Еталон", ha="right", va="center", fontsize=7)
    ax.text(-0.5, 1.5, "Результат", ha="right", va="center", fontsize=7)


def make_figure(rows, title, out_path):
    fig = plt.figure(figsize=(11, 2.1 * len(rows)))
    gs = fig.add_gridspec(2 * len(rows), 1, height_ratios=[1.1, 0.9] * len(rows), hspace=0.35)
    for n, r in enumerate(rows):
        ax_img = fig.add_subplot(gs[2 * n]); ax_txt = fig.add_subplot(gs[2 * n + 1])
        img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
        ax_img.imshow(img, cmap="gray", aspect="equal"); ax_img.axis("off")
        ax_img.set_title(f"{n + 1}) {r.engine}, шум {r.level} %, CER = {r.cer:.3f}", fontsize=8, loc="left")
        draw_diff(ax_txt, r.gt, r.pred)
    fig.suptitle(f"{title}\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10, y=1.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


# %% вибір зразків
df = pd.read_csv(f"{RES_DIR}/metrics_{GROUP}.csv", keep_default_na=False)
df = df[df.config == CONFIG]
df = df.sort_values(["cer", "level"], ascending=False)

worst_all = df.head(N_WORST)                                         # п'ять найгірших за CER
worst_partial = df[df.pred.str.strip() != ""].head(N_WORST)          # п'ять найгірших серед рядків, де щось розпізнано

for name, sub, title in (("all", worst_all, "П'ять найгірших зразків (без обробки)"),
                         ("partial", worst_partial, "П'ять найгірших зразків серед частково розпізнаних")):
    cols = ["engine", "level", "path", "gt", "pred", "cer"]
    out = sub[cols].copy(); out["category"] = ""                      # категорію причини вказує студент
    out.to_csv(f"{RES_DIR}/worst_{GROUP}_{name}.csv", index=False, encoding="utf-8")
    print(f"== {name}\n", out.drop(columns=["path"]).to_string())
    make_figure(list(sub.itertuples()), title, f"{FIG_DIR}/fig9_worst_{GROUP}_{name}.png")
