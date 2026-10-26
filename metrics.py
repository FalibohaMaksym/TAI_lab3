import re
import unicodedata
from rapidfuzz.distance import Levenshtein

APOSTROPHES = "’ʼ‘`´"


def normalize(s, fold=True):
    # NFC, єдиний апостроф, один пробіл, регістр ігноруємо (casefold); пунктуацію НЕ чіпаємо.
    # Причина: еталон SROIE записаний у верхньому регістрі незалежно від надрукованого (див. звіт)
    s = unicodedata.normalize("NFC", str(s))
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold() if fold else s


def cer(gt, pred, fold=True):
    gt, pred = normalize(gt, fold), normalize(pred, fold)
    return Levenshtein.distance(gt, pred) / max(1, len(gt))


def wer(gt, pred, fold=True):
    gt, pred = normalize(gt, fold).split(), normalize(pred, fold).split()
    return Levenshtein.distance(gt, pred) / max(1, len(gt))
