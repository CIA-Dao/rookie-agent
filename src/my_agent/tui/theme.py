from __future__ import annotations

from unicodedata import east_asian_width

_HOT_PINK = "#ff149d"
_ELECTRIC_CYAN = "#00e5ff"
_NEAR_BLACK = "#0d0d0d"
_SURFACE = "#121212"
_PANEL = "#1a1a1a"
_OFF_WHITE = "#f0f0f0"
_DIM = "#a0a0a0"
_WARNING = "#ffbf00"
_ERROR = "#ff4d4f"
_SUCCESS = _ELECTRIC_CYAN
_BRAND_TEXT = "MY AGENT"


def _display_width(text: str) -> int:
    """Return display width: CJK chars count as two columns."""
    return sum(2 if east_asian_width(char) in ("F", "W") else 1 for char in text)


def _soft_wrap_stream_text(text: str, width: int) -> str:
    """Wrap streaming text by display width without inserting Rich markup."""
    if width <= 8:
        return text

    lines: list[str] = []
    current: list[str] = []
    current_width = 0
    for char in text:
        if char == "\n":
            lines.append("".join(current))
            current = []
            current_width = 0
            continue
        char_width = 2 if east_asian_width(char) in ("F", "W") else 1
        if current and current_width + char_width > width:
            lines.append("".join(current))
            current = [char]
            current_width = char_width
        else:
            current.append(char)
            current_width += char_width
    if current or not lines:
        lines.append("".join(current))
    return "\n".join(lines)


def _stream_preview_width(measured_width: int) -> int:
    """Return a stable wrap width while the Textual layout is settling."""
    if measured_width <= 0:
        return 80
    return max(72, measured_width - 16)
