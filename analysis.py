# %% імпорти та налаштування
import os, re, unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
from scipy.stats import spearmanr

from common import STUDENT, VARIANT, SEED, LEVELS

RES_DIR = os.environ.get("OCR_OUT_DIR", "results")
FIG_DIR = os.environ.get("FIG_DIR", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MAIN = ["raw", "median3", "median5", "nlmeans"]        # конфігурації абляції для всіх рівнів шуму
ABL_LEVEL = 5                                         # рівень шуму для абляційної діаграми і boxplot
CER_LIMIT = 0.10                                      # допустимий поріг CER
ENGINES = ["EasyOCR", "PaddleOCR"]
rng_boot = np.random.default_rng(SEED)                # окремий генератор для бутстрепу


def stamp(ax, title):
    """Заголовок графіка з персональним підписом студента."""
    ax.set_title(f"{title}\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)


# %% нормалізація і метрики
from metrics import normalize, cer, wer     # правила нормалізації і формули метрик


def boot_ci(x, n_boot=2000, alpha=0.05):
    """Бутстреп-інтервал для середнього значення метрики."""
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan, np.nan
    means = rng_boot.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def load(name):
    df = pd.read_csv(f"{RES_DIR}/results_{name}.csv", keep_default_na=False)
    df["pred"] = df["pred"].astype(str)
    df["cer"] = [cer(g, p) for g, p in zip(df["gt"], df["pred"])]
    df["wer"] = [wer(g, p) for g, p in zip(df["gt"], df["pred"])]
    df["cer_case_sensitive"] = [cer(g, p, fold=False) for g, p in zip(df["gt"], df["pred"])]   # для порівняння у звіті
    return df


synth, sroie, own = load("synthetic"), load("sroie"), load("own")
for name, df in (("synthetic", synth), ("sroie", sroie), ("own", own)):
    df.to_csv(f"{RES_DIR}/metrics_{name}.csv", index=False, encoding="utf-8")
    print(name, "рядків:", len(df), "| помилок розпізнавання:", int((df["error"] != "").sum()))

# %% вплив рішення про регістр (для звіту): CER без обробки з урахуванням і без урахування регістру
case_effect = (pd.concat([synth.assign(group="synthetic"), sroie.assign(group="sroie"), own.assign(group="own")])
               .query("config == 'raw'")
               .groupby(["group", "engine"])[["cer_case_sensitive", "cer"]].mean().round(3)
               .rename(columns={"cer": "cer_ignore_case"}))
case_effect.to_csv(f"{RES_DIR}/case_effect.csv")
print(case_effect)

# %% таблиця результатів: CER, WER, час (для звіту)
summary = (synth.groupby(["engine", "config", "level"])
           .agg(CER_mean=("cer", "mean"), CER_median=("cer", "median"),
                WER_mean=("wer", "mean"), sec_mean=("sec", "mean"), n=("cer", "size"))
           .round(3))
summary.to_csv(f"{RES_DIR}/summary_synthetic.csv")
print(summary.CER_mean.unstack("level"))

# %% fig1: крива стійкості з бутстреп-ДІ
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, eng in zip(axes, ENGINES):
    for cfg in MAIN:
        g = synth[(synth.engine == eng) & (synth.config == cfg)]
        by_lvl = g.groupby("level")["cer"]
        mean = by_lvl.mean()
        ci = np.array([boot_ci(v) for _, v in by_lvl])
        ax.plot(mean.index, mean.values, marker="o", label=cfg)
        ax.fill_between(mean.index, ci[:, 0], ci[:, 1], alpha=0.15)
    ax.axhline(CER_LIMIT, ls="--", c="grey", lw=1)
    ax.set_xlabel("імпульсний шум «сіль-перець», % пошкоджених пікселів")
    ax.legend()
    stamp(ax, f"Крива стійкості: {eng}")
axes[0].set_ylabel("CER (середнє, 95% ДІ)")
fig.savefig(f"{FIG_DIR}/fig1_robustness.png", dpi=150, bbox_inches="tight")

# %% fig2: boxplot розподілу CER (рівень ABL_LEVEL)
sub = synth[(synth.level == ABL_LEVEL) & synth.config.isin(["raw", "median3"])]
labels, groups = [], []
for eng in ENGINES:
    for cfg in ["raw", "median3"]:
        labels.append(f"{eng}\n{cfg}")
        groups.append(sub[(sub.engine == eng) & (sub.config == cfg)]["cer"].values)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.boxplot(groups, tick_labels=labels, showmeans=True)
for k, g in enumerate(groups, start=1):
    ax.scatter(np.full(len(g), k) + rng_boot.uniform(-0.08, 0.08, len(g)), g, s=10, alpha=0.5)
ax.set_ylabel("CER окремого рядка")
stamp(ax, f"Розподіл CER при шумі {ABL_LEVEL} %")
fig.savefig(f"{FIG_DIR}/fig2_boxplot.png", dpi=150, bbox_inches="tight")

# %% fig3: впевненість і CER (Спірмен) та крива ризик-покриття
raw = synth[synth.config == "raw"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for eng in ENGINES:
    g = raw[raw.engine == eng]
    rho, _ = spearmanr(g.conf, g.cer)
    axes[0].scatter(g.conf, g.cer, s=10, alpha=0.4, label=f"{eng}: ρ = {rho:.2f}")
    g = g.sort_values("conf", ascending=False)
    coverage = np.arange(1, len(g) + 1) / len(g)
    risk = g.cer.cumsum().values / np.arange(1, len(g) + 1)
    axes[1].plot(coverage, risk, label=eng)
axes[0].set_xlabel("впевненість інструмента")
axes[0].set_ylabel("CER")
axes[0].legend()
stamp(axes[0], "Впевненість проти помилки")
axes[1].axhline(0.02, ls="--", c="grey", lw=1)
axes[1].set_xlabel("частка автоматично прийнятих рядків (coverage)")
axes[1].set_ylabel("середній CER прийнятих рядків")
axes[1].legend()
stamp(axes[1], "Крива ризик-покриття")
fig.savefig(f"{FIG_DIR}/fig3_confidence.png", dpi=150, bbox_inches="tight")

# %% fig4: компроміс «час-CER» (усі рівні шуму) і фронт Парето
agg = synth[synth.config.isin(MAIN)].groupby(["engine", "config"]).agg(
    cer=("cer", "mean"), ms=("sec", "mean")).reset_index()
agg["ms"] *= 1000
agg["pareto"] = [not ((agg.ms <= r.ms) & (agg.cer <= r.cer) & ((agg.ms < r.ms) | (agg.cer < r.cer))).any()
                 for r in agg.itertuples()]
fig, ax = plt.subplots(figsize=(7, 4.5))
for r in agg.itertuples():
    ax.scatter(r.ms, r.cer, s=60, c="tab:green" if r.pareto else "grey")
    ax.annotate(f"{r.engine} / {r.config}", (r.ms, r.cer), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("середній час на рядок, мс")
ax.set_ylabel("середній CER (усі рівні шуму)")
stamp(ax, "Компроміс якість-швидкість (зелені: фронт Парето)")
fig.savefig(f"{FIG_DIR}/fig4_pareto.png", dpi=150, bbox_inches="tight")

# %% fig5: абляція попередньої обробки (рівень ABL_LEVEL)
abl = synth[synth.level == ABL_LEVEL].groupby(["config", "engine"])["cer"].mean().unstack("engine")
abl = abl.reindex([c for c in MAIN + ["median3_then_up", "up_then_median3"] if c in abl.index])
ax = abl[ENGINES].plot.barh(figsize=(8, 4.5))
ax.invert_yaxis()
ax.set_xlabel("середній CER")
stamp(ax, f"Внесок кроків попередньої обробки, шум {ABL_LEVEL} %")
ax.figure.savefig(f"{FIG_DIR}/fig5_ablation.png", dpi=150, bbox_inches="tight")
print(abl.round(3))

# %% fig6: найчастіші заміни символів (raw, частково правильні рядки)
def sym(c):
    return "␣" if c == " " else c


def confusions(gt, pred):
    gt, pred = normalize(gt), normalize(pred)
    pairs = []
    for op in Levenshtein.editops(gt, pred):
        if op.tag == "replace":
            pairs.append(f"{sym(gt[op.src_pos])} → {sym(pred[op.dest_pos])}")
        elif op.tag == "delete":
            pairs.append(f"{sym(gt[op.src_pos])} → ∅")
        else:
            pairs.append(f"∅ → {sym(pred[op.dest_pos])}")
    return pairs


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, eng in zip(axes, ENGINES):
    cnt = Counter()
    part = synth[(synth.engine == eng) & (synth.config == "raw") & (synth.cer < 0.5)]
    for r in part.itertuples():
        cnt.update(confusions(r.gt, r.pred))
    top = pd.Series(dict(cnt.most_common(15)))
    if len(top):
        ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("кількість")
    stamp(ax, f"Найчастіші заміни символів ({eng})")
fig.savefig(f"{FIG_DIR}/fig6_confusions.png", dpi=150, bbox_inches="tight")

# %% fig7: порівняння CER на трьох групах даних (raw)
groups = {
    "синтетика (0 %)": synth[(synth.config == "raw") & (synth.level == 0)],
    "SROIE": sroie[sroie.config == "raw"],
    "власні: сторінка": own[(own.config == "raw") & (own.kind == "page")],
    "власні: етикетки": own[(own.config == "raw") & (own.kind == "label")],
}
fig, ax = plt.subplots(figsize=(9, 4.5))
width = 0.38
for j, eng in enumerate(ENGINES):
    means, errs = [], []
    for g in groups.values():
        v = g[g.engine == eng]["cer"].values
        lo, hi = boot_ci(v)
        means.append(v.mean() if len(v) else np.nan)
        errs.append([means[-1] - lo, hi - means[-1]])
    x = np.arange(len(groups)) + (j - 0.5) * width
    ax.bar(x, means, width, yerr=np.array(errs).T, capsize=3, label=eng)
ax.set_xticks(np.arange(len(groups)), list(groups.keys()))
ax.set_ylabel("середній CER (без обробки, 95% ДІ)")
ax.legend()
stamp(ax, "CER на трьох групах даних")
fig.savefig(f"{FIG_DIR}/fig7_groups.png", dpi=150, bbox_inches="tight")

# %% fig8: особливість варіанта: де медіана і NLM перестають допомагати
mean_cer = synth[synth.config.isin(MAIN)].groupby(["engine", "config", "level"])["cer"].mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
gain_rows = []
for ax, eng in zip(axes, ENGINES):
    base = mean_cer[eng]["raw"]
    for cfg in ["median3", "median5", "nlmeans"]:
        gain = base - mean_cer[eng][cfg]               # > 0: фільтр допомагає
        ax.plot(gain.index, gain.values, marker="o", label=cfg)
        # рівень, починаючи з якого фільтр більше не допомагає (gain <= 0 і далі теж)
        stop = None
        for lvl in gain.index[::-1]:
            if gain[lvl] <= 0:
                stop = lvl
            else:
                break
        gain_rows.append({"engine": eng, "filter": cfg, "stops_helping_from_level": stop,
                          **{f"gain_{l}": round(float(gain[l]), 3) for l in gain.index}})
    ax.axhline(0, c="grey", lw=1)
    ax.set_xlabel("імпульсний шум «сіль-перець», %")
    ax.legend()
    stamp(ax, f"Виграш фільтра: CER(raw) − CER(фільтр), {eng}")
axes[0].set_ylabel("виграш у CER (вище 0: фільтр допомагає)")
fig.savefig(f"{FIG_DIR}/fig8_filters_gain.png", dpi=150, bbox_inches="tight")
gains = pd.DataFrame(gain_rows)
gains.to_csv(f"{RES_DIR}/filter_gain.csv", index=False)
print(gains.to_string())

# %% власні дані: таблиця CER по файлах і конфігураціях
print(own.pivot_table(index="file", columns=["engine", "config"], values="cer").round(3).to_string())

# %% п'ять найгірших зразків (синтетика, raw)
cols = ["engine", "level", "gt", "pred", "cer"]
print(synth[synth.config == "raw"].nlargest(5, "cer")[cols].to_string())
