"""Canonical category sections; paths never depend on the caller's cwd."""
import hashlib
import re
from pathlib import Path

GUIDE_PATH = Path(__file__).resolve().parents[1] / "innoglobalhack_avito" / "microcategories-guide.md"


def load_categories(path=GUIDE_PATH):
    text = Path(path).read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## (10[1-6]) (.+)$", text, re.MULTILINE))
    categories = {}
    for index, heading in enumerate(headings):
        mc_id = int(heading[1])
        if mc_id in categories:
            raise ValueError(f"Duplicate guide category: {mc_id}")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        categories[mc_id] = {"name": heading[2].strip(),
                             "section": text[heading.start():end].strip()}
    if set(categories) != set(range(101, 107)):
        raise ValueError("Guide must contain categories 101–106 exactly once")
    return categories


def guide_version(path=GUIDE_PATH):
    return {"file": Path(path).name,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
