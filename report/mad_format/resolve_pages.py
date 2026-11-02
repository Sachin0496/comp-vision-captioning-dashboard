"""Resolve TOC and list page numbers from the rendered PDF."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def page_texts(pdf: Path) -> list[str]:
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read page count")
    count = int(match.group(1))
    pages = []
    for number in range(1, count + 1):
        text = subprocess.run(
            ["pdftotext", "-f", str(number), "-l", str(number), "-layout", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        pages.append(re.sub(r"\s+", " ", text).strip().lower())
    return pages


def first_page(pages: list[str], needle: str, start: int = 0) -> int | None:
    needle = re.sub(r"\s+", " ", needle).strip().lower()
    for index in range(start, len(pages)):
        if needle in pages[index]:
            return index + 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--out", default="page_map.json")
    args = parser.parse_args()
    pdf = Path(args.pdf)
    pages = page_texts(pdf)

    abstract = first_page(pages, "abstract")
    toc = first_page(pages, "table of contents")
    list_page = first_page(pages, "list of tables", (toc or 1))
    abbreviation = first_page(pages, "list of abbreviations", (list_page or 1))
    intro = first_page(pages, "chapter - 1")
    if not all([abstract, toc, list_page, abbreviation, intro]):
        missing = {
            "abstract": abstract,
            "toc": toc,
            "list": list_page,
            "abbrev": abbreviation,
            "intro": intro,
        }
        raise RuntimeError(f"Could not locate front-matter headings: {missing}")

    def roman(number: int) -> str:
        values = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        result = ""
        for value, symbol in values:
            while number >= value:
                result += symbol
                number -= value
        return result

    def front(page: int | None) -> str:
        if page is None:
            return "—"
        return roman(page - abstract + 1)

    def body(page: int | None) -> str:
        if page is None:
            return "—"
        return str(page - intro + 1)

    def after_intro(needle: str, start: int | None = None) -> int | None:
        return first_page(pages, needle, (start or intro) - 1)

    # Locate body headings; TOC occurrences are excluded by starting at `intro`.
    chapter2 = after_intro("chapter 2")
    chapter3 = after_intro("chapter 3")
    chapter4 = after_intro("chapter 4")
    objective = after_intro("1.1 objective")
    overview = after_intro("1.2 overview")
    scope = after_intro("1.3 scope")
    architecture = after_intro("4.1 system architecture diagram")
    results = after_intro("results and discussion")
    conclusion = after_intro("conclusion", results)
    references = after_intro("references", conclusion)

    mapping = {
        "abstract": front(abstract),
        "toc": front(toc),
        "list_tables": front(list_page),
        "list_figures": front(list_page),
        "abbrev": front(abbreviation),
        "intro": body(intro),
        "objective": body(objective),
        "overview": body(overview),
        "scope": body(scope),
        "lit": body(chapter2),
        "problem": body(chapter3),
        "proposed": body(chapter4),
        "architecture": body(architecture),
        "results": body(results),
        "conclusion": body(conclusion),
        "references": body(references),
    }
    for key, label in (
        ("table_4_1", "table 4.1:"),
        ("table_4_2", "table 4.2:"),
        ("table_4_3", "table 4.3:"),
        ("table_5_1", "table 5.1:"),
        ("table_5_2", "table 5.2:"),
        ("table_5_3", "table 5.3:"),
        ("table_5_4", "table 5.4:"),
        ("table_5_5", "table 5.5:"),
        ("figure_4_1", "figure 4.1:"),
        ("figure_5_1", "figure 5.1:"),
        ("figure_5_2", "figure 5.2:"),
        ("figure_5_3", "figure 5.3:"),
        ("figure_5_4", "figure 5.4:"),
    ):
        mapping[key] = body(after_intro(label))

    Path(args.out).write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(mapping, indent=2))


if __name__ == "__main__":
    main()
