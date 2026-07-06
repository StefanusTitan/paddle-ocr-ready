from price_parser import Price


def normalize_amount(raw: str) -> str | None:
    """Best-effort parse of a raw amount string into a plain decimal string.

    Uses price_parser to handle varied formats the LLM might return, e.g.:
      "Rp 1.500.000"  -> "1500000"
      "$12,345.67"     -> "12345.67"
      "150000"         -> "150000"
      "1.500.000,00"   -> "1500000.00"

    Returns the cleaned decimal string, or the original value unchanged
    if parsing fails.
    """
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    price = Price.fromstring(text)

    if price.amount is not None:
        # Format as plain decimal (no scientific notation like "5E+2")
        return f"{price.amount.normalize():f}"

    # price_parser couldn't make sense of it — return as-is
    return text
