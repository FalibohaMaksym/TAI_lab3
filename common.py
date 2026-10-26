import cv2

STUDENT = "Faliboha Maksym"
STUDENT_ID = 981
VARIANT = 15
SEED = 1000 * VARIANT + STUDENT_ID

FACTOR = "saltpepper"
LEVELS = [0, 1, 2, 5, 10, 15]

# конфігурації попередньої обробки для абляції (кроки виконуються в заданому порядку)
CONFIGS = {
    "raw": (),
    "median3": ("median3",),
    "median5": ("median5",),
    "nlmeans": ("nlmeans",),
}
# додаткові конфігурації для перевірки порядку кроків (гіпотеза Н5), лише для цих рівнів шуму
CONFIGS_EXTRA = {
    "median3_then_up": ("median3", "upscale"),
    "up_then_median3": ("upscale", "median3"),
}
EXTRA_LEVELS = [5]


def preprocess(img, steps=()):
    """Попередня обробка: кроки застосовуються в тому порядку, у якому вони перелічені."""
    out = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for step in steps:
        if step == "median3":       # медіанний фільтр 3x3 проти імпульсного шуму
            out = cv2.medianBlur(out, 3)
        elif step == "median5":     # медіанний фільтр 5x5
            out = cv2.medianBlur(out, 5)
        elif step == "nlmeans":     # non-local means
            out = cv2.fastNlMeansDenoising(out, None, h=20, templateWindowSize=7, searchWindowSize=21)
        elif step == "upscale":     # збільшення вдвічі
            out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError(f"невідомий крок обробки: {step}")
    return out
