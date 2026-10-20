"""Shared JLM branding: author, version, title art and palette (no heavy imports)."""

AUTHOR = "jddd"
VERSION = "v0.3 · 4B"

# "JLM" in the DOS Rebel figlet font (generated once with pyfiglet, ░ shading swapped for ▒), embedded verbatim.
TITLE_ART = [
    "       █████ █████       ██████   ██████",
    "      ▒▒███ ▒▒███       ▒▒██████ ██████",
    "       ▒███  ▒███        ▒███▒█████▒███",
    "       ▒███  ▒███        ▒███▒▒███ ▒███",
    "       ▒███  ▒███        ▒███ ▒▒▒  ▒███",
    " ███   ▒███  ▒███      █ ▒███      ▒███",
    "▒▒████████   ███████████ █████     █████",
    " ▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒     ▒▒▒▒▒",
]
SUBTITLE = "Jiahao Language Model"
BANNER_COLORS = ["#FFE066", "#FFD43B", "#FCC419", "#FAB005", "#F59F00", "#F08C00", "#E8590C", "#D9480F"]
SHADE_COLOR = "#7A4A00"

GOLD = "#F59F00"


def prepare_console() -> None:
    """Switch a Windows console to UTF-8 so ▒/█/emoji never crash on a GBK code page."""
    import os
    import sys

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
