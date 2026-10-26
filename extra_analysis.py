# Питання 2 і 3 підрозділу 7.3: ширина довірчих інтервалів і бутстреп парної різниці CER
# %% імпорти та налаштування
import os, json
import numpy as np
import pandas as pd

from common import SEED, LEVELS
from metrics import cer

RES_DIR = os.environ.get("OCR_OUT_DIR", "results")
EXTRA_LEVEL = int(os.environ.get("EXTRA_LEVEL", "15"))
N_BOOT = 2000
rng = np.random.default_rng(SEED)


def boot_means(x, n_boot=N_BOOT):
    x = np.asarray(x)
    return rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)


def ci(x, alpha=0.05):
    lo, hi = np.quantile(boot_means(x), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_diff_ci(a, b, alpha=0.05):
    """Бутстреп різниці середніх CER (a - b) на тих самих рядках: індекси рядків вибираються спільно."""
    d = np.asarray(a) - np.asarray(b)
    lo, hi = np.quantile(boot_means(d), [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


# %% дані: основний набір (raw) і додатковий
main = pd.read_csv(f"{RES_DIR}/results_synthetic.csv", keep_default_na=False)
main = main[main.config == "raw"].copy()
main["cer"] = [cer(g, p) for g, p in zip(main["gt"], main["pred"])]
extra = pd.read_csv(f"{RES_DIR}/results_extra_level{EXTRA_LEVEL}.csv", keep_default_na=False)
extra["cer"] = [cer(g, p) for g, p in zip(extra["gt"], extra["pred"])]

out = {}

# %% питання 3: рівень з найбільшою різницею між інструментами і бутстреп парної різниці
piv = main.pivot_table(index=["level", "id"], columns="engine", values="cer")
mean_diff = (piv["EasyOCR"] - piv["PaddleOCR"]).groupby("level").mean()
worst_gap_level = int(mean_diff.abs().idxmax())
a = piv.loc[worst_gap_level]["EasyOCR"].values
b = piv.loc[worst_gap_level]["PaddleOCR"].values
d, lo, hi = paired_diff_ci(a, b)
out["q3"] = {"level_max_gap": worst_gap_level, "mean_diff_by_level": mean_diff.round(3).to_dict(),
             "n": int(len(a)), "diff_EasyOCR_minus_PaddleOCR": round(d, 3),
             "ci95": [round(lo, 3), round(hi, 3)], "contains_zero": bool(lo <= 0 <= hi)}

# %% питання 2: ширина інтервалу на найгіршому рівні для кожного інструмента
out["q2"] = {}
for eng in ("EasyOCR", "PaddleOCR"):
    orig = main[(main.engine == eng) & (main.level == EXTRA_LEVEL)]["cer"].values      # 40 рядків
    new = extra[extra.engine == eng]["cer"].values
    both = np.concatenate([orig, new])
    lo0, hi0 = ci(orig)
    lo1, hi1 = ci(both)
    w0, w1 = hi0 - lo0, hi1 - lo1
    out["q2"][eng] = {
        "n_original": int(len(orig)), "n_combined": int(len(both)),
        "mean_original": round(float(orig.mean()), 3), "ci_original": [round(lo0, 3), round(hi0, 3)],
        "width_original": round(w0, 3),
        "mean_combined": round(float(both.mean()), 3), "ci_combined": [round(lo1, 3), round(hi1, 3)],
        "width_combined": round(w1, 3),
        "width_ratio_original_to_combined": round(w0 / w1, 2),
        "theory_ratio_sqrt_n": round(float(np.sqrt(len(both) / len(orig))), 2),
    }

# %% парна різниця на збільшеній вибірці (той самий рівень)
e = extra.pivot_table(index="id", columns="engine", values="cer")
a2 = np.concatenate([main[(main.engine == "EasyOCR") & (main.level == EXTRA_LEVEL)].sort_values("id")["cer"].values,
                     e["EasyOCR"].values])
b2 = np.concatenate([main[(main.engine == "PaddleOCR") & (main.level == EXTRA_LEVEL)].sort_values("id")["cer"].values,
                     e["PaddleOCR"].values])
d2, lo2, hi2 = paired_diff_ci(a2, b2)
out["q3_extended"] = {"level": EXTRA_LEVEL, "n": int(len(a2)), "diff_EasyOCR_minus_PaddleOCR": round(d2, 3),
                      "ci95": [round(lo2, 3), round(hi2, 3)], "contains_zero": bool(lo2 <= 0 <= hi2)}

with open(f"{RES_DIR}/extra_analysis.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
