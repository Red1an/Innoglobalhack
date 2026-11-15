"""Черновики для колонки descriptions: предложения исходного текста по теме категории, без LLM."""
import re

NAMES = {101: "Ремонт под ключ", 102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника",
         105: "Электрика", 106: "Поклейка обоев и малярные работы"}
CLASS_WORDS = {
    102: r"штукатур|оштукатур|маяк|венецианк|короед|травертин",
    103: r"плит|кафел|керамогранит|мозаик|затирк|шв\w* плитк|фартук",
    104: r"сантех|водопровод|канализ|труб|отоплен|радиатор|смесител|унитаз|инсталляц|бойлер|водонагрев|полотенцесуш|ванн|душев",
    105: r"электр|проводк|розет|выключат|щит|освещен|светильник|люстр|кабел|автомат|узо",
    106: r"обо[иеяю]|покрас|окрас|маляр|шпакл|шпатл|стеклохолст|грунт|шлиф|краск",
}
CLASS_RE = {c: re.compile(p, re.I) for c, p in CLASS_WORDS.items()}
CONTACT = re.compile(r"\+?\d[\d\s()-]{8,}\d|https?://|www\.|@\w|t\.me|whatsapp|телеграм|вотсап|звоните", re.I)
SEO = re.compile(r"#\w|также ищут|ключевые слова|теги|по запросам", re.I)
VAGUE = re.compile(r"этих видов|из перечисленн|любой из|все виды|под ключ|комплексн", re.I)
SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text):
    return [s.strip(" •-–—*\t") for s in SPLIT.split(text) if len(s.strip(" •-–—*\t")) > 2]


def _end(s):
    return s if s[-1] in ".!?…" else s + "."


def extract(description, target, max_general=3):
    topical, general = [], []
    for s in sentences(description):
        if CONTACT.search(s) or SEO.search(s):
            continue
        cats = {c for c, rx in CLASS_RE.items() if rx.search(s)}
        if target in cats:
            topical.append(s)
        elif not cats and not VAGUE.search(s) and len(s) <= 200:
            general.append(s)
    if not topical:
        topical = [f"{NAMES[target]}."]
    return " ".join(_end(s) for s in topical + general[:max_general])[:1200]
