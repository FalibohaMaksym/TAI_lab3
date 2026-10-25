import re

raw = open("https://uabook.com.ua/wp-content/uploads/2023/02/vovchok-marko-instytutka457.txt", "rb").read()

# кодування таких файлів буває різним
for enc in ("utf-8-sig", "cp1251"):
    try:
        text = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue

text = text.replace("\r", "\n")
text = re.sub(r"\s+", " ", text)
text = text.replace("\u2019", "'").replace("\u02bc", "'").replace("`", "'")

# ділимо на речення за . ! ? …
parts = re.split(r"(?<=[.!?…])\s+", text)

# довгі речення додатково ріжемо за комами
sentences = []
for p in parts:
    if len(p) <= 70:
        sentences.append(p)
    else:
        sentences += re.split(r"(?<=,)\s+", p)

# залишаємо підходящі за довжиною, без латиниці й цифр
corpus = [s.strip(" —-\"«»") for s in sentences]
corpus = [s for s in corpus if 20 <= len(s) <= 70 and not re.search(r"[A-Za-z0-9]", s)]
corpus = list(dict.fromkeys(corpus))   # прибрати дублікати

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus))

print("Рядків у corpus.txt:", len(corpus))