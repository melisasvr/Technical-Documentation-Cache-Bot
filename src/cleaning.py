"""
Clean text extracted from PDFs so it wastes fewer tokens.

Many PDFs (especially ones saved from a web browser) repeat the same header or
footer on every page, e.g. "10/16/26, 12:42 PM  RFC 8259 ... | RFC Editor".
That noise costs tokens and adds nothing, so we remove it.
Pure Python (no LangChain imports) so it is easy to test on its own.
"""
import re
from collections import Counter

# Browser "print to PDF" footers look like: "https://site/page#x 12/33 <real text...>"
_URL_PAGE_PREFIX = re.compile(r"^https?://\S+\s+\d+/\d+\s?")

# Table-of-contents dot leaders: "Introduction . . . . . . . . 3" -> "Introduction 3"
_DOT_LEADERS = re.compile(r"(?:\s?\.){4,}")


def _normalize(line: str) -> str:
    """Make 'Page 3' and 'Page 4' look identical so repeated footers match."""
    return re.sub(r"\d+", "#", line.strip())


def clean_pages(pages: list[str], min_repeats: int = 3, min_fraction: float = 0.3) -> list[str]:
    """
    Remove header/footer lines that repeat across many pages, strip indentation,
    and collapse blank lines. Returns one cleaned string per input page.
    """
    # 1) Strip browser URL/page-counter prefixes and leading/trailing spaces
    split_pages = []
    for text in pages:
        lines = [
            _DOT_LEADERS.sub(" ", _URL_PAGE_PREFIX.sub("", ln)).strip()
            for ln in text.splitlines()
        ]
        split_pages.append(lines)

    # 2) Count on how many pages each (normalized) line appears
    counts: Counter[str] = Counter()
    for lines in split_pages:
        counts.update({_normalize(ln) for ln in lines if ln})

    threshold = max(min_repeats, int(len(pages) * min_fraction))
    repeated = {k for k, v in counts.items() if v >= threshold} if len(pages) > min_repeats else set()

    # 3) Drop repeated lines, collapse empty lines
    cleaned = []
    for lines in split_pages:
        kept = [ln for ln in lines if ln and _normalize(ln) not in repeated]
        cleaned.append("\n".join(kept))
    return cleaned