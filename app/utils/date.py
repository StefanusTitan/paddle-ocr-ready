import re
from datetime import datetime

# Indonesian month names for date parsing
_INDO_MONTHS = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
    "september": 9, "oktober": 10, "november": 11, "desember": 12,
    # Common English month names as fallback
    "january": 1, "february": 2, "march": 3, "may": 5,
    "june": 6, "july": 7, "august": 8, "october": 10, "december": 12,
    # Abbreviated
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "agu": 8, "ags": 8, "aug": 8,
    "sep": 9, "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12,
}


def normalize_date(raw: str) -> str:
    """Best-effort parse of a raw date string into YYYY-MM-DD.

    Handles common receipt formats:
      - DD/MM/YYYY or DD-MM-YYYY
      - YYYY-MM-DD or YYYY/MM/DD (ISO-ish)
      - DD Month YYYY (Indonesian or English month names)
      - DD.MM.YYYY

    Returns the original string unchanged if parsing fails.
    """
    s = raw.strip()

    # Strip trailing time component (e.g. "15:00", "14:30:22")
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?$", "", s)

    # 1. Try "DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY"
    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. Try "YYYY-MM-DD" or "YYYY/MM/DD" (already ISO-like)
    m = re.match(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$", s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 3. Try "DD Month YYYY" or "DD Month, YYYY" (Indonesian/English)
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$", s)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        year = int(m.group(3))
        month = _INDO_MONTHS.get(month_str)
        if month:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # 4. Try "Month DD, YYYY" (English-style)
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", s)
    if m:
        month_str = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3))
        month = _INDO_MONTHS.get(month_str)
        if month:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass

    return s  # Return unchanged if no pattern matched
