"""Консервативные правила для разметки конкретных работ в полном description."""

import re

FLAGS = re.IGNORECASE
SEO = re.compile(
    r'(?:^|[\n.!?]|(?<=\s))(?:возможно[, ]+вы искали|также ищут|часто ищут|'
    r'ключевые слова(?: для поиска)?|для поиска|поисковые запросы|поисковые теги|'
    r'вы (?:могли )?искать нас по|вы могли найти наше объявление, потому что искали|'
    r'нас (?:также )?ищут(?:\s+(?:по таким запросам|со словами))?|нас можно\s+'
    r'(?:(?:так\s*же\s+)?найти)(?:\s+по этим запросам)?|нас находят по запросам|'
    r'наши запросы|ключевые запросы(?: для поиска)?|для удобства поиска|'
    r'seo[ -]?запросы|теги)[* \t]*(?::|\n)', FLAGS,
)
SEO_LIST = re.compile(
    r'^(?:ремонт|отделк\w+|черновая отделка|внутренние отделочные работы|'
    r'мастер|шпаклёвка|шпаклевка|шпатлевка|поклейка)[^\n.!?]{5,90}[,; /]',
    FLAGS,
)
PLASTER = r'\b(?:[а-яё]*штукатур\w*|маяк\w*|венецианк\w*|рот[ -]?банд\w*|короед\w*|микроцемент\w*|терразит\w*|лепнин\w*|мокр\w+ фасад\w*)'
LEVEL = r'\b(?:выравн\w*|выровн\w*|ровня\w*)[^\n.;]{0,30}?(?:стен|потол|поверх)\w*|\bш(?:пакл|патл)\w*'
TILE = r'(?:плитк\w*|кафел\w*|керамогранит\w*|мозаик\w*)'
PLUMB = (
    r'(?:сантехник\w*|санитарн\w+\s+(?:техник\w*|оборудован\w*)|водопровод\w*|'
    r'водоснабжен\w*|канализац\w*|(?:водяных?\s+)?труб\w*|'
    r'смесител\w*|унитаз\w*|раковин\w*|сифон\w*|инсталляц\w*|'
    r'ванн(?:ы|у)\b|душев\w+\s+(?:кабин\w*|систем\w*)|радиатор\w*|полотенцесушител\w*)'
)
ELECTRIC = (
    r'(?:электрик\w*|электросет\w*|электрощит\w*|электроточ\w*|'
    r'(?:электро)?проводк\w*|кабел\w*|розет\w*|подрозетник\w*|выключател\w*|'
    r'электропровод\w*|узо\b|дифавтомат\w*|светильник\w*|люстр\w*|освещени\w*)'
)
PAINT = r'(?:покраск\w*|окраск\w*|(?:по|о)?крашив\w*|крас(?:им|ит|ить|ят)|покрас(?:им|ить|ит|ят))'
WALLPAPER = r'(?:обои\b|обоев\w*|обоя\w*|стеклохолст\w*)'
FIT = r'(?:монтаж\w*|установ\w+|замен\w+|подключ\w+|провед\w*|развод\w+|проклад\w+|перен\w+|диагностик\w*)'
PREP = re.compile(
    rf'\b(?:под|для|к)\s+(?:будущ\w+\s+|самостоятельн\w+\s+)?'
    rf'(?:{PAINT}|{WALLPAPER}|краск\w*|(?:по|о)клей\w*\s+обоев|{TILE}|укладк\w*\s+{TILE})'
    rf'(?:[ \t]*(?:(?:или|и)\s+|/\s*(?:под\s+)?|,\s*под\s+)'
    rf'(?:{PAINT}|{WALLPAPER}|краск\w*|(?:по|о)клей\w*\s+обоев|{TILE}))*', FLAGS,
)
EXTERIOR_TILE = r'(?:тротуарн\w*|потолочн\w*)\s+плитк\w*'
SERVICE = {
    102: (
        r'\b(?:штукатур(?:ка|ку|им|ить|кой)|(?:о|за|пере)штукатур\w*|'
        r'штукатурн\w+\s+работ\w*|венецианк\w*|'
        r'(?:выравн\w*|выровн\w*)[^\n.;]{0,35}(?:стен\w*|потолк\w*)[^\n.;]{0,25}по маякам)\b',
    ),
    103: (
        rf'\b(?:уклад\w+|укладыва\w+|клад[её]м|ложим|монтаж\w*|'
        rf'замен\w+|облицо\w+|запил\w*|полож\w+)[^\n.;:—]{{0,66}}?\b{TILE}',
        r'\bплиточн\w+\s+работ\w*|\bзатир\w*\s+(?:плиточн\w+\s+)?шв\w*|'
        r'\bзапил\w*\s+под\s+45',
        r'\bчистов\w+\s+отделк\w*\s*[:—–(]\s*[^\n.;]{0,78}\b(?:плитка|керамогранит)\b',
    ),
    104: (
        rf'\b{FIT}[^\n.;:—]{{0,48}}?\b{PLUMB}|'
        r'\bремонт\w*\s+(?:смесител\w*|унитаз\w*|душев\w+\s+кабин\w*)',
        r'\bсантехническ\w+\s+(?:работ\w*|монтаж\w*)|\bразвод\w+\s+(?:холодн\w+\s+и\s+горяч\w+\s+)?(?:водоснабжен\w*|воды)|'
        r'\b(?:устран\w+\s+протеч\w*|прочист\w+\s+засор\w*|реставр\w+\s+ванн\w*)|'
        r'\bэлектромонтаж\w*\s+и\s+сантехник[аиу]\b|'
        r'^\s*(?:[—–•*-]\s*)?сантехник[аиу]\s*(?:[.;]|$)|'
        r'\b(?:устройств\w*|монтаж\w*|установ\w+)\s+водян\w*\s+т[её]пл\w*\s+пол\w*',
    ),
    105: (
        rf'\b{FIT}[^\n.;:—]{{0,44}}?\b{ELECTRIC}|\bремонт\w*\s+{ELECTRIC}',
        r'\bэлектро\s*монтаж\w*|\bэлектрическ\w+\s+работ\w*|'
        r'^\s*(?:[—–•*-]\s*)?электрик[аиу]\s*(?:[.;]|$)|'
        r'\bштроб\w*\s+(?:стен\w*\s+)?под\s+проводк\w*|'
        r'\b(?:устройств\w*|монтаж\w*|установ\w+)\s+электрическ\w*\s+т[её]пл\w*\s+пол\w*',
    ),
    106: (
        rf'\b{PAINT}\b|\bм[ао]лярн\w+\s+работ\w*',
        rf'\b(?:покле\w+|окле\w+|клеим|накле\w+|замен\w+|перекле\w+)[^\n.;:—]{{0,28}}?\b{WALLPAPER}',
        r'\b(?:моющ\w+\s+)?краска\s+на\s+(?:стен\w*|потолк\w*)',
    ),
}
WEAK = {
    102: PLASTER,
    103: rf'\b(?:{TILE}|плиточ\w*|затир\w*|запил\w*|платки\b)',
    104: rf'\b(?:{PLUMB}|сантех\w*|ванн\w*|отоплен\w*|т[её]пл\w+\s+пол\w*|бойлер\w*|водонагрев\w*|протеч\w*|засор\w*|пос[уy]домоеч\w*|стиральн\w*|канализац\w*)',
    105: rf'\b(?:{ELECTRIC}|электр\w*|электро\s*монтаж\w*|подсвет\w*|свет\w*|штроб\w*|слаботоч\w*|т[её]пл\w+\s+пол\w*)',
    106: rf'\b(?:{PAINT}|{WALLPAPER}|м[ао]ляр\w*|краск\w*|(?:по|о)клей\w*)',
}
REQUEST = re.compile(r'^\s*(?:(?:требуется|ищу|ищем|нужен|нужна|нужны)\s+(?:ремонт|мастер\w*|бригад\w*|отделочни\w*)|я\s*алиса)', FLAGS)
HASHTAG = re.compile(r'(?<!\w)#[а-яё][а-яё_0-9]+', FLAGS)
TAG_PATTERNS = {
    102: re.compile(r'штукатурка(?:стен|потолков|откосов)?', FLAGS),
    103: re.compile(r'(?:укладка|монтаж)(?:плитки|керамогранита|кафеля)|плиточныеработы', FLAGS),
    104: re.compile(r'(?:монтаж|установка|замена)сантехники|сантехническиеработы', FLAGS),
    105: re.compile(r'электромонтаж|заменапроводки|монтажрозеток', FLAGS),
    106: re.compile(r'(?:поклейка|оклейка)обоев|(?:покраска|окраска)(?:стен|потолков)', FLAGS),
}
TAG_CONTRADICTION = re.compile(
    r'\b(?:не\s+(?:предлага\w*|делаем|выполня\w*|бер[её]м\w*|занима\w*)|'
    r'кроме|сторонн\w+|самостоятельно|только|подготов\w*|проектирован\w*)\b', FLAGS,
)
OFFER = re.compile(
    r'\b(?:выполня\w+|выполним|предлага\w+|занима\w+|делаем|сделаем|работаем|'
    r'бер[её]м|входит|включает|услуги|что (?:делаем|выполняем|входит)|перечень работ|'
    r'наши работы|финишная отделка|этапы работ|комплекс работ|'
    r'укладываем|клад[её]м|монтируем|подключаем|устанавливаем|штукатурим|красим)\b', FLAGS,
)
NEG = re.compile(
    r'\b(?:не\s+(?:бер[её]м\w*|выполня\w*|делаем|занима\w*|предлага\w*|оказыва\w*|'
    r'входит|входят|уклад\w*|монтиру\w*|работаем|красим|клеим|штукатурим)|кроме|без)\b', FLAGS,
)
OTHER = re.compile(
    r'\b(?:подготов\w*\s+(?:под|для)|демонтаж\w*|сняти\w*|удалени\w*|'
    r'заказчик\w*\s+сам|сторонн\w+\s+(?:специалист|мастер|подрядчик)|'
    r'порекоменд\w+|рекомендуем\s+знаком\w+|есть специалисты|не\s+бер[её]мся|'
    r'технадзор|пример\s+(?:объекта|сметы)|риск\w*|ошибк\w*|'
    r'продаж\w+|в\s+наличии|подбор\w*\s+материал\w*)\b', FLAGS,
)
PROJECT = re.compile(r'\b(?:проектирован\w*|проект|схем\w*|подбор)\s+[^\n.;]{0,25}$', FLAGS)
ABSTRACT = re.compile(r'(?:не просто|не про|после\s+|вместо\s+|перед\s+|ошибк\w+\s+(?:в|при)\s+)\s*[«"]?\s*$', FLAGS)
MATERIAL_PAINT = re.compile(r'(?:моющ\w+\s+)?краска\s+на\s+', FLAGS)
RULES = {key: tuple(re.compile(pattern, FLAGS | re.M) for pattern in patterns) for key, patterns in SERVICE.items()}
SIGNALS = {key: re.compile(pattern, FLAGS) for key, pattern in WEAK.items()}
EXTERIOR = re.compile(EXTERIOR_TILE, FLAGS)
LEVELING = re.compile(LEVEL, FLAGS)
CLAUSE = re.compile(r'[\n;!?]|(?<!\d)\.(?!\d)')
ROOF = re.compile(r'\b(?:кровельн\w+\s+работ\w*|ремонт\w*\s+крыш\w*|монтаж\w*\s+кровл\w*)', FLAGS)
MATERIALS = re.compile(
    r'^\s*(?:работа\w+\s+с\s+материалами|материалы,?\s+с\s+которыми\s+работа\w+)\s*:\s*\n'
    r'(?:\s*[-–—*•✓✔➤▪][^\n]*(?:\n|$))+', FLAGS | re.M,
)
PREPARATION_RESULT = re.compile(
    r'\b(?:под|для)\s+(?:\w+[, ]+){0,6}или\s+' + TILE, FLAGS,
)
PLUMBING_WORK = re.compile(
    r'\b(?:сантех\w*|труб\w*|канализац\w*|смесител\w*|подключ\w*|водоснабжен\w*|'
    r'радиатор\w*|отоплен\w*|реставр\w*|полотенцесушител\w*|водонагрев\w*|'
    r'унитаз\w*|раковин\w*|душев\w*|т[её]пл\w+\s+пол\w*)', FLAGS,
)
ROOM_FINISH = re.compile(r'\b(?:монтаж|укладка|облицовка)\s+(?:плитки|кафеля|панелей)[^\n.]{0,40}\b(?:ванн\w*|санузл\w*)', FLAGS)
PAINT_OFFER = re.compile(r'(?im)^\s*(?:[—–•*-]\s*)?(?:покраска|окраска|шпакл[её]вка)\s+(?:стен|потолков)')
ELECTRIC_DESIGN = re.compile(r'\b(?:проектирован\w*|проект\w*)\s+(?:электрик\w*|электроснабжен\w*|электросет\w*)', FLAGS)
PREP_TASK = re.compile(
    rf'\b(?:подготов\w*|ш(?:пакл|патл)\w*|выравн\w*|штукатур\w*|грунтов\w*|шлифов\w*)'
    rf'[^\n.;]{{0,65}}?(?:{PREP.pattern})', FLAGS,
)
OUTSIDE_ONLY_103 = re.compile(r'\b(?:уклад\w*|укладыва\w*|монтаж\w*)\s+' + EXTERIOR_TILE, FLAGS)
TILE_PREP_ONLY = re.compile(
    r'(?:под\s+(?:укладк\w+\s+)?|(?:демонтаж\w*|сняти\w+)\s+(?:стар\w+\s+)?)' + TILE, FLAGS,
)
GENERAL_REASONS = {
    102: 'Штукатурные работы явно указаны',
    103: 'Действие относится к плитке, в том числе в списке материалов или работ',
    104: 'Прямое предложение сантехнических работ',
    105: 'Прямое предложение электромонтажных работ',
    106: 'Явно предложена покраска, поклейка или результат финишной отделки',
}


def mask(pattern, text):
    return pattern.sub(lambda match: ' ' * len(match.group()), text)


def description_body(text):
    marker = SEO.search(text)
    end = marker.start() if marker else len(text)
    for paragraph in reversed(list(re.finditer(r'[^\n]+', text[:end]))):
        value = paragraph.group().strip()
        if len(value) < 180 or not SEO_LIST.match(value):
            continue
        separators = sum(value.count(symbol) for symbol in (',', ' / ', ';'))
        if separators >= 7 and not re.search(r'[.!?]\s+[А-ЯЁ]|\b(?:выполняем|предлагаем|делаем)\b', value):
            end = paragraph.start()
            break
    return text[:end]


def supported(category, original, active, start, stop):
    """Проверяем роль фразы в текущем предложении или пункте списка."""
    left = CLAUSE.split(original[max(0, start - 95):start])[-1]
    right = CLAUSE.split(original[stop:stop + 65])[0]
    fragment = active[start:stop]
    # Отрицание отдельного заказа срабатывает только около соответствующей фразы.
    if re.search(r'\bне\s+(?:бер[её]м\w*|выполня\w*)\s+(?:отдельн\w+\s+)?заказ', left, FLAGS):
        return False
    if re.search(r'\bне\s+(?:дела\w*|выполня\w*|бер[её]м\w*)\s*:?\s*(?:частичн\w*|отдельн\w*|за\s+отдельн\w*)[^)]*\(', left, FLAGS):
        return False
    if re.search(r'^(?:(?![.:]).){0,55}\bне\s+(?:выполня\w*|делаем|укладыва\w*|занима\w*|бер[её]м\w*|принима\w*|входит|входят)\b', right, FLAGS):
        return False
    local = re.split(r'[,/]|\s[—–]\s', left)[-1]
    if NEG.search(local) or ABSTRACT.search(local):
        return False
    if OTHER.search(local) and not OFFER.search(local):
        return False
    if re.search(r'\b(?:правильность|проверяю|проверяем|не\s+хотите\s+искать)\b', left, FLAGS):
        return False
    if re.search(r'\b(?:если|когда)\s+[^,]{0,75}$', left, FLAGS) and not OFFER.search(local):
        return False
    if re.search(r'\b(?:помож\w*|совет\w*|покаж\w*|обсуд\w*)\s+[^,]{0,32}$', local, FLAGS):
        return False
    if re.search(r'\bне\s+нуж\w*\s+[^,]{0,40}$', local, FLAGS):
        return False
    if category == 102 and re.search(r'\b(?:старая|старую|старой)\s*$', local, FLAGS):
        return False
    if category == 103 and EXTERIOR.search(original[start:stop]):
        return False
    if category == 104:
        if re.search(r'\b(?:крыш\w*|кровл\w*|фасад\w*|дымоход\w*|вентиляц\w*)', fragment + right[:26], FLAGS):
            return False
        if 'протеч' in fragment.casefold() and ROOF.search(original[:250]):
            return False
    if category == 105 and PROJECT.search(local):
        return False
    if category == 106:
        if re.search(r'\bне просто(?:\s+\w+){0,3}\s*$', local, FLAGS):
            return False
        if MATERIAL_PAINT.match(fragment):
            if not re.search(r'\b(?:отделка|отделку|финиш)\b', left, FLAGS):
                return False
        if re.match(r'м[ао]лярн', fragment, FLAGS) and re.match(r'\s*[:—–]\s*только\b', right, FLAGS):
            return False
    return True


def evidence(category, text, active):
    for pattern in RULES[category]:
        for match in pattern.finditer(active):
            if supported(category, text, active, *match.span()):
                return text[match.start():match.end()]
    return ''


def corroborated_tag(category, description, body):
    if TAG_CONTRADICTION.search(body) or not SIGNALS[category].search(body):
        return ''
    for tag in HASHTAG.finditer(description):
        if TAG_PATTERNS[category].fullmatch(tag.group()[1:].replace('_', '')):
            return tag.group()
    return ''


def relabel(description, labels):
    """Возвращает предложения 0/1 с точными цитатами из исходного текста."""
    body = description_body(description)
    active = mask(PREP, mask(MATERIALS, body))
    if REQUEST.match(body):
        return {}
    has_offer = bool(OFFER.search(body))
    decisions = {}
    for category in RULES:
        current = category in labels
        text = mask(EXTERIOR, active) if category == 103 else active
        proof = evidence(category, body, text)
        tag = corroborated_tag(category, description, body) if has_offer and not proof else ''
        if proof or tag:
            if not current and has_offer:
                reason = GENERAL_REASONS[category] if proof else 'Конкретный тег подтверждает услугу по основному тексту'
                decisions[category] = (1, proof or tag, reason)
            continue

        full_signal = SIGNALS[category].search(mask(MATERIALS, description))
        if not current:
            continue

        if category == 102 and not full_signal:
            near = LEVELING.search(description)
            if near:
                decisions[category] = (0, near.group(), 'Выравнивание без указания штукатурки или только шпаклёвка')
        elif category == 103:
            preparation = TILE_PREP_ONLY.search(body)
            outside = OUTSIDE_ONLY_103.search(body)
            # Сигнал вне этих выражений может быть предложением кафеля или укладки.
            unmatched = mask(OUTSIDE_ONLY_103, mask(TILE_PREP_ONLY, mask(PREPARATION_RESULT, description)))
            if (preparation or outside) and not SIGNALS[103].search(unmatched):
                decisions[category] = (0, (preparation or outside).group(), 'Только подготовка, демонтаж или наружная плитка')
        elif category == 104:
            room_finish = ROOM_FINISH.search(body)
            if room_finish and not PLUMBING_WORK.search(description):
                decisions[category] = (0, room_finish.group(), 'Описана отделка ванной, сантехнические работы не указаны')
        elif category == 105:
            paint_offer = PAINT_OFFER.search(body)
            design = ELECTRIC_DESIGN.search(body)
            if design and not SIGNALS[105].search(mask(ELECTRIC_DESIGN, description)):
                decisions[category] = (0, design.group(), 'Электрика только в проекте, монтаж не предложен')
            elif paint_offer and not full_signal:
                decisions[category] = (0, paint_offer.group().strip(), 'В описании малярные услуги, электромонтаж не указан')
        elif category == 106:
            preparation = PREP_TASK.search(body)
            unmasked = mask(PREP, description)
            if preparation and not SIGNALS[106].search(unmasked):
                decisions[category] = (0, preparation.group(), 'Покраска или поклейка названа как цель подготовки')
    return decisions
