import re
import dateparser

# A few Indonesian abbreviations dateparser's 'id' locale doesn't recognize on its own
_EXTRA_MONTHS_ID = {
    "ags": "agustus", "ag": "agustus",
    "des": "desember",
    "jan": "januari", "feb": "februari", "mar": "maret", "apr": "april",
    "jun": "juni", "jul": "juli", "sep": "september", "okt": "oktober", "nov": "november",
}


def normalize_date(raw: str) -> str:
    """Best-effort parse of a raw date string into YYYY-MM-DD using dateparser.

    Handles DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, ISO YYYY-MM-DD, and
    'DD Month YYYY' in both Indonesian and English, including a few
    Indonesian month abbreviations dateparser misses on its own (e.g. 'Ags').

    Returns the original string unchanged if parsing fails.
    """
    s = raw.strip()
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?$", "", s)  # strip trailing time

    # expand IDN abbreviations dateparser doesn't know
    s = re.sub(r"[A-Za-z]+", lambda m: _EXTRA_MONTHS_ID.get(m.group(0).lower(), m.group(0)), s)

    # only force day-first ordering when the string ISN'T already year-first (ISO-ish),
    # otherwise dateparser mangles '2024-05-12' into treating it as day-first
    settings = {"STRICT_PARSING": False}
    if not re.match(r"^\d{4}[/-]", s):
        settings["DATE_ORDER"] = "DMY"

    dt = dateparser.parse(s, languages=["id", "en"], settings=settings)
    return dt.strftime("%Y-%m-%d") if dt else raw