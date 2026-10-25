import os
import numpy as np
import pandas as pd
from datasets import load_dataset

VARIANT = 15
STUDENT_ID = 981
SEED = 1000 * VARIANT + STUDENT_ID

DATASET = "priyank-m/SROIE_2019_text_recognition"
SPLIT = "test"
N_SAMPLES = 120

hf = load_dataset(DATASET, split=SPLIT)
print(hf)
print(hf.features)

os.makedirs("data/sroie", exist_ok=True)

# окремий генератор, щоб вибір не залежав від rng синтетичного набору
rng_open = np.random.default_rng(SEED)
indices = sorted(int(i) for i in rng_open.choice(len(hf), size=N_SAMPLES, replace=False))

records = []
for idx in indices:
    sample = hf[idx]
    text = sample["text"].strip()
    if not text:
        continue
    path = f"data/sroie/sroie_{SPLIT}_{idx:05d}.png"
    sample["image"].save(path)
    records.append({"split": SPLIT, "idx": idx, "path": path, "gt": text})

ids = pd.DataFrame(records)
ids.to_csv("data/sroie_ids.csv", index=False, encoding="utf-8")
print("Вибрано:", N_SAMPLES, "| збережено:", len(ids), "| відсіяно порожніх:", N_SAMPLES - len(ids))

print(ids.head())
