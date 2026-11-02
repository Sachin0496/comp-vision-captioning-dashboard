"""Build the project report in the KPR / MAD report format.

The layout intentionally mirrors the supplied PDF template:
cover -> bonafide certificate -> acknowledgement -> abstract -> TOC ->
list of tables/figures -> abbreviations -> Chapters 1-5 -> conclusion ->
references.  Personal fields are left as placeholders for the student to fill.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DEFAULT_OUT = ROOT / "MAD_Report_Image_Captioning.docx"
PAGE_MAP_PATH = ROOT / "page_map.json"

FONT = "Times New Roman"
BODY_PT = 14
SMALL_PT = 10.5
CAPTION_PT = 10.5
BLACK = RGBColor(0, 0, 0)
RED = RGBColor(0xC0, 0x00, 0x00)
MAROON = RGBColor(0x7B, 0x1F, 0x2B)

TITLE = "Image Captioning with CNN-LSTM and Attention: Measuring Object Hallucination"

TOC_ROWS = [
    ("", "ABSTRACT", "abstract"),
    ("", "LIST OF TABLES", "list_tables"),
    ("", "LIST OF FIGURES", "list_figures"),
    ("", "LIST OF ABBREVIATIONS", "abbrev"),
    ("1", "INTRODUCTION", "intro"),
    ("", "1.1 OBJECTIVE", "objective"),
    ("", "1.2 OVERVIEW", "overview"),
    ("", "1.3 SCOPE", "scope"),
    ("2", "LITERATURE SURVEY", "lit"),
    ("3", "PROBLEM IDENTIFICATION", "problem"),
    ("4", "PROPOSED SYSTEM", "proposed"),
    ("", "4.1 SYSTEM ARCHITECTURE DIAGRAM", "architecture"),
    ("5", "RESULT AND DISCUSSION", "results"),
    ("7", "CONCLUSION", "conclusion"),
    ("8", "REFERENCES", "references"),
]

TABLE_ROWS = [
    ("4.1", "DATASET SPLITS", "table_4_1"),
    ("4.2", "CAPTIONING MODEL VARIANTS", "table_4_2"),
    ("4.3", "TRAINING HYPERPARAMETERS", "table_4_3"),
    ("5.1", "CAPTION-QUALITY RESULTS", "table_5_1"),
    ("5.2", "OBJECT-HALLUCINATION RESULTS", "table_5_2"),
    ("5.3", "PAIRED MODEL COMPARISONS", "table_5_3"),
    ("5.4", "BLEU-HALLUCINATION CORRELATIONS", "table_5_4"),
    ("5.5", "MATCHED-BLEU HALLUCINATION RATES", "table_5_5"),
]

FIGURE_ROWS = [
    ("4.1", "SYSTEM ARCHITECTURE", "figure_4_1"),
    ("5.1", "MODEL COMPARISON", "figure_5_1"),
    ("5.2", "BLEU-4 VERSUS HALLUCINATED OBJECTS", "figure_5_2"),
    ("5.3", "WORD-LEVEL ATTENTION MAPS", "figure_5_3"),
    ("5.4", "QUALITATIVE MODEL COMPARISON", "figure_5_4"),
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def set_run_font(run, size: float | None = None, bold: bool | None = None, italic: bool | None = None,
                 color: RGBColor | None = None, name: str = FONT) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def set_cell_margins(cell, top=40, start=80, bottom=40, end=80) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    mar = tcPr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tcPr.append(mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, visible=True, size=8) -> None:
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single" if visible else "nil")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
        borders.append(element)
    tblPr.append(borders)


def set_repeat_table_header(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def prevent_row_split(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    cantSplit = OxmlElement("w:cantSplit")
    trPr.append(cantSplit)


def set_fixed_layout(table) -> None:
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)


def set_col_widths(table, widths) -> None:
    for row in table.rows:
        for index, width in enumerate(widths):
            row.cells[index].width = Inches(width)
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)


def set_page_number_format(section, fmt: str, start: int = 1) -> None:
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    pgNumType = OxmlElement("w:pgNumType")
    pgNumType.set(qn("w:fmt"), fmt)
    pgNumType.set(qn("w:start"), str(start))
    sectPr.append(pgNumType)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    set_run_font(run, 11, name="Calibri")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instr, separate, text, end):
        run._r.append(element)


def add_footer(section, fmt: str) -> None:
    footer = section.footer
    footer.is_linked_to_previous = False
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    add_page_field(paragraph)
    set_page_number_format(section, fmt, 1)


def configure_section(section) -> None:
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)


def para(text="", *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=BODY_PT, bold=False, italic=False,
         color=None, first_line=0.0, left_indent=0.0, before=0, after=6, spacing=1.5,
         keep_next=False, keep_together=False, page_break=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Inches(first_line)
    pf.left_indent = Inches(left_indent)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    pf.keep_with_next = keep_next
    pf.keep_together = keep_together
    pf.page_break_before = page_break
    if text:
        run = p.add_run(text)
        set_run_font(run, size, bold=bold, italic=italic, color=color)
    return p


def rich_para(parts, **kwargs):
    p = para("", **kwargs)
    for part in parts:
        if isinstance(part, tuple):
            text = part[0]
            bold = part[1] if len(part) > 1 else False
            italic = part[2] if len(part) > 2 else False
            color = part[3] if len(part) > 3 else None
        else:
            text, bold, italic, color = part, False, False, None
        run = p.add_run(text)
        set_run_font(run, kwargs.get("size", BODY_PT), bold=bold, italic=italic, color=color)
    return p


def body(text, **kwargs):
    kwargs.setdefault("first_line", 0.5)
    return para(text, **kwargs)


def body_rich(parts, **kwargs):
    kwargs.setdefault("first_line", 0.5)
    return rich_para(parts, **kwargs)


def section_heading(text, *, page_break=False):
    p = para(
        text,
        align=WD_ALIGN_PARAGRAPH.LEFT,
        size=14,
        bold=True,
        before=12,
        after=6,
        spacing=1.0,
        keep_next=True,
        page_break=page_break,
    )
    return p


def chapter_heading(lines, *, page_break=True):
    first = True
    for index, line in enumerate(lines):
        p = para(
            line,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            size=14,
            bold=True,
            before=0 if index == 0 else 0,
            after=18 if index == len(lines) - 1 else 0,
            spacing=1.0,
            keep_next=True,
            page_break=(page_break and first),
        )
        first = False
        if index < len(lines) - 1:
            p.paragraph_format.keep_with_next = True


def add_table(headers, rows, widths, caption, *, font_size=10.5, header_size=10.5,
              align_center=False, border=True, page_key=None, after=10):
    cap = para(
        caption,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=CAPTION_PT,
        bold=True,
        before=6,
        after=3,
        spacing=1.0,
        keep_next=True,
    )
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_fixed_layout(table)
    set_table_borders(table, visible=border)
    header = table.rows[0]
    for index, value in enumerate(headers):
        cell = header.cells[index]
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(str(value))
        set_run_font(run, header_size, bold=True)
        set_cell_margins(cell)
    set_repeat_table_header(header)
    prevent_row_split(header)
    for row_values in rows:
        row = table.add_row()
        prevent_row_split(row)
        for index, value in enumerate(row_values):
            cell = row.cells[index]
            cell.text = ""
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (align_center and index > 0) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(value))
            set_run_font(run, font_size)
            set_cell_margins(cell)
    set_col_widths(table, widths)
    # Keep the table on one page when possible: every row but the last asks to
    # stay with the next row.  This prevents orphaned headers on a new page.
    for row in table.rows[:-1]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True
    para("", size=2, after=after, spacing=1.0)
    return table


def add_figure(filename, width, caption, *, page_break=False):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    run.add_picture(str(FIG / filename), width=Inches(width))
    para(
        caption,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=CAPTION_PT,
        bold=False,
        italic=False,
        before=0,
        after=8,
        spacing=1.0,
    )


def blank(height=6):
    """Exact-height spacer paragraph (empty paragraphs otherwise inherit 14 pt)."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = Pt(max(1, height))
    run = p.add_run("")
    set_run_font(run, 1)
    return p


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------
def cover_page():
    # Logo + title table, matching the template's top block.
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_fixed_layout(table)
    set_table_borders(table, visible=False)
    widths = [1.05, 4.4, 1.05]
    set_col_widths(table, widths)
    for cell in table.rows[0].cells:
        set_cell_margins(cell, top=0, start=0, bottom=0, end=0)
    logo_cell = table.rows[0].cells[0]
    logo_p = logo_cell.paragraphs[0]
    logo_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    logo_p.paragraph_format.space_before = Pt(0)
    logo_p.paragraph_format.space_after = Pt(0)
    logo_p.add_run().add_picture(str(FIG / "kpr_emblem.png"), width=Inches(0.95))
    title_cell = table.rows[0].cells[1]
    title_p = title_cell.paragraphs[0]
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(0)
    title_p.paragraph_format.line_spacing = 1.0
    run = title_p.add_run(TITLE)
    set_run_font(run, 16, bold=True)
    table.rows[0].height = Inches(0.9)

    blank(2)
    para("U21CS503 Mobile Application Development", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=0, spacing=1.0)
    blank(4)
    para("REPORT", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=0, spacing=1.0)
    blank(18)
    para("Submitted by", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, italic=True, after=0, spacing=1.0)
    blank(16)

    student = doc.add_table(rows=1, cols=2)
    student.alignment = WD_TABLE_ALIGNMENT.CENTER
    student.autofit = False
    set_fixed_layout(student)
    set_table_borders(student, visible=False)
    set_col_widths(student, [2.9, 1.7])
    for cell in student.rows[0].cells:
        set_cell_margins(cell, top=0, start=0, bottom=0, end=0)
    left = student.rows[0].cells[0].paragraphs[0]
    left.alignment = WD_ALIGN_PARAGRAPH.LEFT
    left.paragraph_format.space_after = Pt(0)
    run = left.add_run("<Student Name>")
    set_run_font(run, 15, bold=True)
    right = student.rows[0].cells[1].paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right.paragraph_format.space_after = Pt(0)
    run = right.add_run("<Roll Number>")
    set_run_font(run, 15, bold=True)

    blank(24)
    para("In partial fulfillment for the award of the degree", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, italic=True, after=0, spacing=1.0)
    blank(8)
    para("of", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, italic=True, after=0, spacing=1.0)
    blank(8)
    para("BACHELOR OF ENGINEERING", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, after=0, spacing=1.0)
    blank(8)
    para("in", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, italic=True, after=0, spacing=1.0)
    blank(8)
    para("COMPUTER SCIENCE AND ENGINEERING", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, after=0, spacing=1.0)
    blank(26)
    para("KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=0, spacing=1.0)
    para("(Autonomous Institution)", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=0, spacing=1.0)
    blank(22)
    para("ARASUR, COIMBATORE 641 407", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=0, spacing=1.0)
    para("ANNA UNIVERSITY: CHENNAI 600 025", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, after=0, spacing=1.0)
    blank(4)
    para("OCTOBER 2026", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, after=0, spacing=1.0)
    doc.add_page_break()


def certificate_page():
    para("BONAFIDE CERTIFICATE", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=24, spacing=1.0)
    rich_para(
        [
            ("Certified that this project report ", False, False, None),
            (f"“{TITLE}”", True, False, RED),
            (" is the bonafide work of ", False, False, None),
        ],
        first_line=0.0,
        after=8,
    )
    rich_para(
        [
            ("“<Student Name 1> (<Roll Number>)          <Student Name 2> (<Roll Number>)          <Student Name 3> (<Roll Number>)”", True, False, None),
            (" who carried out the project work under my supervision.", False, False, None),
        ],
        first_line=0.0,
        after=42,
    )
    # Guide block.
    for line, bold in (
        ("FACULTY INCHARGE", True),
        ("Mr. T. Munirathinam", True),
        ("Assistant Professor III", True),
        ("Department of Computer Science and Engineering", True),
        ("KPR Institute of Engineering and Technology,", True),
        ("Coimbatore", True),
    ):
        para(line, align=WD_ALIGN_PARAGRAPH.LEFT, size=12.5, bold=bold, after=0, spacing=1.15)
    blank(80)
    para("Submitted for Project Viva-Voce held on ______________________", align=WD_ALIGN_PARAGRAPH.LEFT, size=12.5, bold=True, after=0, spacing=1.0)
    doc.add_page_break()


def acknowledgement_page():
    para("ACKNOWLEDGEMENT", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=18, spacing=1.0)
    paragraphs = [
        [
            ("We wish to express our heartfelt thanks and gratitude to our honorable Chairman ", False),
            ("Dr. K. P. RAMASAMY", True),
            (", KPR Group of Institutions, for providing the facilities during the course of our study in the college.", False),
        ],
        [
            ("We would like to express our deep gratitude and sincere regards to our beloved and honourable Principal ", False),
            ("Dr. Akila Muthuramalingam", True),
            (", KPR Institute of Engineering and Technology, who gave us the opportunity to frame the project to the full satisfaction.", False),
        ],
        [
            ("We owe a genuine gratitude to ", False),
            ("Dr. MIRUNA JOE AMALI S", True),
            (", Professor, Head of the Department, Department of Computer Science and Engineering, for his valuable support and encouragement during this project work.", False),
        ],
        [
            ("We are grateful to ", False),
            ("Mr. Munirathinam T", True),
            (", Assistant Professor III, Department of Computer Science and Engineering, the project supervisor, for his timely suggestions and constant encouragement and support that led to the accomplishment of the project.", False),
        ],
        [
            ("We owe special thanks to Year head ", False),
            ("Dr. PRIYA V", True),
            (", Associate Professor, Department of Computer Science and Engineering, for her continuous support and guidance throughout the project.", False),
        ],
        [
            ("The acknowledgement would be incomplete without a word of thanks to all our parents, faculty members, supporting staff and friends for their continuous support and sincere help throughout our project.", False),
        ],
    ]
    for parts in paragraphs:
        rich_para([(text, bold, False, None) for text, bold in parts], first_line=0.5, after=8)
    doc.add_page_break()


def abstract_page():
    para("ABSTRACT", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=12, spacing=1.0)
    text = (
        "Image captioning systems are usually assessed with BLEU and related n-gram overlap metrics. "
        "These scores reward similarity to reference sentences but do not reveal whether a generated caption names an object that is absent from the image. "
        "This failure, called object hallucination, is important because a fluent caption can be confidently wrong. "
        "This project presents a controlled, laptop-scale study of object hallucination using 5,000 images from the COCO 2017 validation set, divided into 4,000 training, 500 validation and 500 test images. "
        "A frozen ResNet-18 encoder supplies a global image vector and a 7×7 spatial feature grid. Three captioners share the same LSTM decoder and training budget: a global-feature baseline, a Bahdanau attention model, and an object-aware model that also receives score-weighted labels from a COCO-pretrained Faster R-CNN detector. "
        "BLEU-1–4, CIDEr-D and CHAIR-style hallucination rates are measured on the held-out split, with paired bootstrap confidence intervals and McNemar tests. "
        "The object-aware model gives the best BLEU-4 (18.03) and the lowest CHAIRs (23.64%), compared with 16.32/34.01% for attention and 11.94/49.32% for the global baseline. "
        "Sentence BLEU-4 and hallucinated-object count have a moderate negative Spearman correlation (ρ = −0.327; image-cluster p < 0.001), but at matched BLEU levels the models still differ by up to 33 percentage points in hallucination rate. "
        "The results show that grounding metrics reveal errors that BLEU cannot, and that object-label fusion is a practical way to improve both caption quality and visual grounding."
    )
    body(text, first_line=0.0, after=0)
    doc.add_page_break()


def toc_page(page_map):
    para("TABLE OF CONTENTS", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=12, spacing=1.0)
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_fixed_layout(table)
    set_table_borders(table, visible=True, size=8)
    widths = [0.9, 4.4, 1.2]
    set_col_widths(table, widths)
    headers = ["CHAPTER\nNO", "TITLE", "PAGE NO"]
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(text)
        set_run_font(run, 10.5, bold=True)
        set_cell_margins(cell, top=50, bottom=50)
    set_repeat_table_header(table.rows[0])
    prevent_row_split(table.rows[0])
    for number, title, key in TOC_ROWS:
        row = table.add_row()
        prevent_row_split(row)
        values = [number, title, page_map.get(key, "00")]
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.text = ""
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i != 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(value))
            set_run_font(run, 10.5, bold=(title in {"INTRODUCTION", "LITERATURE SURVEY", "PROBLEM IDENTIFICATION", "PROPOSED SYSTEM", "RESULT AND DISCUSSION", "CONCLUSION", "REFERENCES"}))
            set_cell_margins(cell, top=55, bottom=55)
    set_col_widths(table, widths)
    doc.add_page_break()


def list_page(page_map):
    para("LIST OF TABLES", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=12, spacing=1.0)
    table = doc.add_table(rows=0, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_fixed_layout(table)
    set_table_borders(table, visible=False)
    widths = [1.0, 4.3, 1.2]
    set_col_widths(table, widths)
    for number, title, key in TABLE_ROWS:
        row = table.add_row()
        prevent_row_split(row)
        values = [number, title, page_map.get(key, "00")]
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.text = ""
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i != 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(value))
            set_run_font(run, 11)
            set_cell_margins(cell, top=45, bottom=45, start=20, end=20)
    set_col_widths(table, widths)
    blank(18)
    para("LIST OF FIGURES", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=12, spacing=1.0)
    table = doc.add_table(rows=0, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_fixed_layout(table)
    set_table_borders(table, visible=False)
    widths = [0.8, 4.5, 1.2]
    set_col_widths(table, widths)
    for number, title, key in FIGURE_ROWS:
        row = table.add_row()
        prevent_row_split(row)
        values = [number, title, page_map.get(key, "00")]
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.text = ""
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i != 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(value))
            set_run_font(run, 11)
            set_cell_margins(cell, top=45, bottom=45, start=20, end=20)
    set_col_widths(table, widths)
    doc.add_page_break()


def abbreviations_page():
    para("LIST OF ABBREVIATIONS", align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, before=0, after=18, spacing=1.0)
    items = [
        ("BLEU", "Bilingual Evaluation Understudy"),
        ("CHAIR", "Caption Hallucination Assessment with Image Relevance"),
        ("CIDEr-D", "Consensus-based Image Description Evaluation (D)"),
        ("CNN", "Convolutional Neural Network"),
        ("COCO", "Common Objects in Context"),
        ("FPN", "Feature Pyramid Network"),
        ("LSTM", "Long Short-Term Memory"),
        ("MPS", "Metal Performance Shaders"),
        ("R-CNN", "Region-based Convolutional Neural Network"),
        ("TF-IDF", "Term Frequency–Inverse Document Frequency"),
    ]
    for abbreviation, expansion in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(f"{abbreviation} - {expansion}")
        set_run_font(run, 13)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------
def chapter_1():
    chapter_heading(["CHAPTER - 1", "INTRODUCTION"], page_break=False)
    section_heading("1.1 Objective", page_break=False)
    body(
        "The objective of this project is to build and evaluate a compact image-captioning system that can be trained and analysed on a single laptop, while exposing a failure mode that conventional caption metrics hide. The work has four specific aims:"
    )
    for text in (
        "1. Construct a reproducible COCO captioning benchmark with training, validation and held-out test splits, together with ground-truth object annotations for every image.",
        "2. Implement three controlled CNN–LSTM captioners: a global-feature baseline, a Bahdanau attention model, and an object-aware model that receives detected object labels.",
        "3. Evaluate each model with BLEU-1–4 and CIDEr-D, and with CHAIR-style object-hallucination metrics that compare caption mentions against COCO instance annotations.",
        "4. Analyse whether better BLEU scores correspond to fewer hallucinations, using caption-level correlations, BLEU quartiles and matched-BLEU comparisons.",
    ):
        p = para(text, first_line=0.0, left_indent=0.3, after=4)
        p.paragraph_format.first_line_indent = Inches(-0.3)
    body(
        "The project deliberately uses a small, controlled setting rather than a state-of-the-art captioning model. All three variants share the same visual features, decoder architecture, optimiser, training budget and decoding procedure, so differences in BLEU and hallucination can be attributed to the information available to the decoder rather than to unrelated changes in capacity or training data."
    )
    body(
        "A second objective is transparency. The complete pipeline — dataset preparation, feature extraction, detector labelling, training, evaluation and figure generation — is implemented as a reproducible codebase. The report documents the design choices and the limitations of the benchmark so that the results can be interpreted honestly."
    )

    section_heading("1.2 Overview")
    body(
        "Image captioning is the task of generating a natural-language description of an image. A typical neural captioner encodes the image with a convolutional neural network (CNN) and decodes the resulting representation with a recurrent neural network (RNN), usually an LSTM. The encoder may produce a single global vector, as in the Show-and-Tell model, or a spatial grid that a decoder can attend over, as in Show-Attend-and-Tell. Training maximises the likelihood of reference captions and evaluation usually compares generated and reference sentences with n-gram overlap metrics such as BLEU or CIDEr."
    )
    body(
        "This evaluation paradigm has a blind spot. A caption can be fluent, grammatical and similar to a reference while naming an object that is not in the image. For example, a model may describe a kitchen as containing an oven when only a microwave is present, or add a person to a photograph of an empty room. Such errors are called object hallucinations. They matter in assistive technology, image search and safety-critical applications, where a confident but false description can mislead a user."
    )
    body(
        "The project studies hallucination with three models of increasing visual access. The global baseline receives only a single image vector. The attention model receives a 7×7 grid and learns where to look at each decoding step. The object-aware model additionally receives labels and confidence scores from an off-the-shelf object detector, which are embedded and supplied to the decoder at every step. The comparison asks whether attention and explicit object evidence improve grounding, and whether BLEU reflects that improvement."
    )
    body(
        "The dataset is a 5,000-image re-split of COCO 2017 validation images. COCO provides five reference captions and instance annotations for 80 object categories, which makes it suitable for both captioning and CHAIR-style evaluation. The images are divided into 4,000 training, 500 validation and 500 test images. CNN features are extracted once and cached, allowing all three models to be trained on a CPU without approaching the memory limit of the test machine."
    )

    section_heading("1.3 Scope")
    body(
        "The scope of this project is a controlled, small-scale investigation rather than a state-of-the-art captioning system. The models use a frozen ResNet-18 backbone and a single-layer LSTM decoder with 512 hidden units. The object-aware model uses detector outputs, not ground-truth object annotations, so detector false positives and missed detections are part of the treatment. Captions are generated greedily, and the evaluation uses one training run per model."
    )
    body(
        "The benchmark is restricted to the 80 COCO object classes. Nouns outside those classes, such as sky, grass or water, are not scored as objects. A visible object that is missing from the COCO annotations is counted as a hallucination, which is a known limitation of CHAIR and is discussed in the results. The project does not attempt human evaluation, reinforcement learning, beam search, large-scale pretraining, or training on the full COCO train2017 split."
    )
    body(
        "Within these boundaries, the report provides a complete and reproducible comparison of global, attention-based and object-aware captioning. It contributes a working pipeline, an analysis of hallucination behaviour, attention visualisations, and evidence about the relationship between BLEU and grounding. The limitations are stated explicitly so that the conclusions are not generalised beyond the measured setting."
    )


def chapter_2():
    chapter_heading(["CHAPTER 2", "LITERATURE SURVEY"])
    body(
        "Image captioning has developed from retrieval-based pipelines to end-to-end neural systems. Early work generated captions by retrieving similar sentences or by filling templates from detected objects, attributes and relationships. These methods produced readable descriptions but were limited by hand-designed rules and by the coverage of the retrieval database. The neural encoder–decoder formulation replaced this pipeline with a single differentiable model trained on image–sentence pairs."
    )
    body_rich(
        [
            ("CNN–LSTM captioning. ", True),
            ("Vinyals et al. (2015) introduced Show and Tell, which encodes an image with a CNN and feeds the resulting vector into an LSTM as its initial state. The model is trained with teacher forcing to maximise the likelihood of each reference caption. Karpathy and Fei-Fei (2015) aligned sentence fragments with image regions, while Mao et al. (2015) used a multimodal recurrent network. These models established the encoder–decoder template that remains the basis of modern captioners. Their limitation is that a single global vector must summarise all visual content before decoding begins.", False),
        ]
    )
    body_rich(
        [
            ("Visual attention. ", True),
            ("Xu et al. (2015) introduced Show, Attend and Tell, which computes a attention distribution over a spatial feature grid at every decoding step. The decoder receives a context vector formed by a weighted sum of image locations, allowing different words to focus on different regions. Lu et al. (2017) added a visual sentinel so that the model can decide when to attend and when to rely on language context. Anderson et al. (2018) replaced grid features with bottom-up region proposals and top-down attention, improving both captioning and visual question answering. Attention improves interpretability, but it is trained only through the caption likelihood objective; nothing forces the attention map to correspond to the object being named.", False),
        ]
    )
    body_rich(
        [
            ("Object hallucination. ", True),
            ("Rohrbach et al. (2018) systematically studied captions that mention objects not present in the image and introduced the CHAIR metrics. CHAIRs measures the fraction of mentioned objects that are hallucinated, while CHAIRi measures the fraction of captions containing at least one hallucinated object. Their analysis showed that models can score well on standard metrics while generating object errors, and that hallucination is not simply a consequence of poor fluency. Hendricks et al. (2018) showed related biases in gender and activity descriptions, demonstrating that caption models can reproduce and amplify corpus regularities.", False),
        ]
    )
    body_rich(
        [
            ("Grounded and object-aware captioning. ", True),
            ("Several lines of work inject explicit object or region information into the decoder. You et al. (2016) used semantic attention over detected attributes and objects. Cornia et al. (2019) proposed a controllable and grounded captioning framework that conditions on regions and objects. Biten et al. (2019) used entity context for news image captioning. More recent visual–language pretraining methods learn grounded representations from large image–text corpora, but they require substantially more data and compute than are available in this project. Detector-label fusion is a lightweight alternative: it gives the decoder object vocabulary and confidence evidence without a large pretrained caption model.", False),
        ]
    )
    body_rich(
        [
            ("Evaluation metrics. ", True),
            ("BLEU (Papineni et al., 2002) measures clipped n-gram precision with a brevity penalty. CIDEr (Vedantam et al., 2015) uses TF-IDF weighted n-grams and a length penalty to reward consensus with multiple references. SPICE (Anderson et al., 2016) parses captions into scene graphs and compares objects, attributes and relations, which is closer to grounding than n-gram overlap. CHAIR (Rohrbach et al., 2018) directly checks object mentions against annotations. Each metric captures a different aspect of quality; none is a complete measure of caption correctness.", False),
        ]
    )
    body(
        "A recurring question is whether better caption metrics imply better grounding. BLEU and CIDEr reward overlap with references, and references themselves may omit or add details. A model can increase its score by producing safe, high-frequency phrases that are partially correct. CHAIR adds a complementary view, but it depends on the completeness of object annotations and on synonym matching. The literature therefore motivates controlled comparisons that report both quality and grounding metrics rather than treating one as a proxy for the other."
    )
    body(
        "The present work differs from prior studies in its controlled, laptop-scale design. Instead of comparing published systems trained with different backbones and data, it holds the visual encoder, decoder, optimiser and training budget fixed while changing only the information available to the decoder. This makes the effect of attention and detected object labels easier to isolate. It also makes the benchmark reproducible without specialised hardware, at the cost of lower absolute scores and a smaller training set."
    )
    body(
        "The research gap addressed here is threefold. First, many captioning comparisons change several components at once, so the contribution of attention or object labels is difficult to separate. Second, hallucination is often reported as a single number without an analysis of how it varies with caption quality. Third, object-label fusion is usually evaluated with oracle labels or large pretrained models, leaving open the question of whether noisy detector outputs help in a small model. This project addresses all three points with a single controlled experiment."
    )


def chapter_3():
    chapter_heading(["CHAPTER 3", "PROBLEM IDENTIFICATION AND METHODOLOGY"])
    section_heading("3.1 Problem Identification", page_break=False)
    body(
        "The central problem is that the objective used to train and rank captioning models is not the objective that users care about. A captioner is optimised to assign high probability to reference words, and it is evaluated by n-gram overlap with those references. Neither process directly penalises a caption for mentioning an object that is absent from the image. A model can therefore become more fluent and more reference-like while becoming less faithful to the visual evidence."
    )
    body(
        "This problem is visible in the outputs of the baseline model. The global-feature captioner frequently produces generic descriptions such as “a man is standing on a street” or “a plate of food with a sandwich and a fork”, even when the image contains a different scene. The caption is syntactically valid and contains common COCO vocabulary, but it may introduce a person, a table or a food item that is not present. Such errors are difficult for a human reader to detect without seeing the image, and they are invisible to corpus BLEU."
    )
    body(
        "Attention is often assumed to solve this problem because it provides a spatial interface between words and image regions. However, attention is learned through the same likelihood objective. A model can learn a plausible attention map and still choose a familiar object word, especially when the training set is small. Attention may improve the average quality of captions without guaranteeing that every mentioned object is grounded. This motivates an evaluation that measures both attention behaviour and object-level correctness."
    )
    body(
        "A second candidate remedy is to provide the decoder with detected object labels. This gives the model explicit evidence about which objects are present and can discourage it from inventing others. The remedy is imperfect: an off-the-shelf detector produces false positives and misses small or occluded objects, and the captioner may copy those mistakes. The key question is whether the benefit of additional object evidence outweighs the noise. Answering it requires a controlled comparison in which only the object input changes."
    )
    rich_para(
        [
            ("Problem statement. ", True),
            ("Given a captioning model that is trained and ranked with n-gram overlap metrics, does better BLEU imply fewer object hallucinations, and can visual attention or detected object labels reduce hallucination without sacrificing caption quality? The project answers this question by measuring BLEU, CIDEr-D and CHAIR on a held-out split for three controlled model variants.", False),
        ],
        first_line=0.0,
        after=10,
    )
    body("The investigation is guided by four research questions:")
    for text in (
        "RQ1. Can a compact CNN–LSTM captioner be trained reproducibly on a laptop-scale COCO split and produce meaningful captions?",
        "RQ2. Does adding spatial attention improve BLEU and reduce object hallucination relative to a global-feature baseline?",
        "RQ3. Does adding noisy detector object labels further reduce hallucination, and at what cost or benefit to caption quality?",
        "RQ4. Is higher BLEU associated with fewer hallucinations at the caption level, or can models with similar BLEU differ substantially in grounding?",
    ):
        p = para(text, first_line=0.0, left_indent=0.3, after=6)
        p.paragraph_format.first_line_indent = Inches(-0.3)
    body(
        "The expected contribution is not a new state-of-the-art architecture, but a clear measurement. If attention and object labels reduce hallucination while improving BLEU, that supports the use of explicit visual evidence. If BLEU and hallucination diverge, that demonstrates the need for grounding-aware evaluation. Either outcome is informative, provided the comparison is controlled and the limitations are reported."
    )


def chapter_4():
    chapter_heading(["CHAPTER 4", "PROPOSED SYSTEM"])
    body(
        "The proposed system is a modular captioning pipeline with three interchangeable decoders. Figure 4.1 shows the data flow. Images are encoded once by a frozen ResNet-18. The spatial feature grid supports attention, the global vector initialises the LSTM state, and an independent detector supplies object labels that are embedded and fused into the decoder. The same evaluation code is applied to all variants."
    )
    section_heading("4.1 System Architecture Diagram", page_break=False)
    add_figure(
        "architecture.png",
        6.2,
        "Figure 4.1: System architecture. The visual encoder supplies global and spatial features; the detector supplies object labels; the LSTM decoder produces a caption that is evaluated for quality and grounding.",
    )
    body(
        "The architecture separates feature extraction from sequence decoding. This separation is deliberate: the CNN and detector are run once per image and their outputs are cached, so training the LSTM does not require repeated forward passes through large networks. On a 16 GB laptop, this reduces memory use and makes the three-model comparison practical. The cache also guarantees that every model sees exactly the same visual and object evidence."
    )
    body_rich(
        [
            ("Dataset and preprocessing. ", True),
            ("The benchmark uses a deterministic re-split of COCO 2017 validation images. COCO provides five reference captions per image and instance annotations for 80 object categories. The split sizes are given in Table 4.1. The training vocabulary is built only from training captions, with a minimum frequency of three; it contains 2,916 entries, of which 2,912 are lexical word types and four are special tokens. Images are resized to 256 pixels on the shorter side and centre-cropped to 224×224 for the ResNet encoder.", False),
        ]
    )
    add_table(
        ["Split", "Images", "Captions"],
        [["Training", "4,000", "20,010"], ["Validation", "500", "2,503"], ["Test", "500", "2,501"]],
        [2.0, 1.6, 1.8],
        "Table 4.1: Dataset splits used in the experiment.",
        align_center=True,
    )
    body_rich(
        [
            ("Visual feature extraction. ", True),
            ("A ResNet-18 pretrained on ImageNet is used as a frozen encoder. The final convolutional stage produces a 7×7 grid of 512-dimensional vectors; global average pooling of the same grid gives a 512-dimensional image vector. Features are stored as float16 arrays and memory-mapped during training. The encoder is never fine-tuned, which keeps the comparison focused on the decoder and prevents the CNN from overfitting the small training split.", False),
        ]
    )
    body_rich(
        [
            ("Captioning models. ", True),
            ("All variants use a single-layer LSTM with 512 hidden units, 256-dimensional word embeddings and dropout of 0.3. The global baseline initialises the LSTM hidden and cell states from the image vector and then receives only the previous word embedding. The attention model adds additive attention over the 49 spatial locations: the previous hidden state and each location are projected, combined through a tanh, and scored to produce a softmax distribution. The weighted sum of locations forms a context vector that is concatenated with the word embedding. The object-aware model additionally embeds each detected label, weights it by detector confidence, averages the embeddings, and concatenates the resulting object vector with the LSTM input at every step. Table 4.2 summarises the variants.", False),
        ]
    )
    add_table(
        ["Variant", "Visual input", "Attention", "Object labels", "Parameters"],
        [
            ["Global", "512-d image vector", "No", "No", "4.34 M"],
            ["Attention", "7×7×512 grid", "Bahdanau", "No", "5.92 M"],
            ["Object-aware", "7×7×512 grid", "Bahdanau", "Yes (detector)", "6.19 M"],
        ],
        [1.45, 1.75, 1.05, 1.25, 1.0],
        "Table 4.2: Captioning model variants and parameter counts.",
        font_size=10,
        align_center=True,
    )
    body_rich(
        [
            ("Object-label fusion. ", True),
            ("Object labels come from a Faster R-CNN with a MobileNetV3-320 backbone, pretrained on COCO. The detector is run on all images at a confidence threshold of 0.30, keeping at most 12 detections. Each class index maps to a 128-dimensional embedding. For an image with detections (label i, score s_i), the object vector is the score-weighted mean of the label embeddings. Images with no detections receive a zero vector. The detector is independent of the caption models and never sees test annotations; its false positives and missed detections are part of the experimental treatment.", False),
        ]
    )
    body_rich(
        [
            ("Training protocol. ", True),
            ("All models are trained with AdamW and a OneCycle learning-rate schedule for 12 epochs. Each epoch samples 8,000 caption–image pairs from the 20,010-pair training pool, which exposes the models to a broad range of captions without the cost of a full pass over all five references. The loss is cross-entropy with label smoothing 0.05 and padding ignored. Gradients are clipped at 5.0. Checkpoints are selected by validation loss rather than validation BLEU, because a BLEU-selected checkpoint can be a degenerate high-frequency captioner. The full configuration is listed in Table 4.3.", False),
        ]
    )
    add_table(
        ["Parameter", "Value"],
        [
            ["Optimiser", "AdamW"],
            ["Learning rate", "1 × 10⁻³ (OneCycle)"],
            ["Weight decay", "1 × 10⁻⁴"],
            ["Batch size", "96"],
            ["Epochs", "12"],
            ["Sampled pairs per epoch", "8,000"],
            ["Dropout", "0.3"],
            ["Label smoothing", "0.05"],
            ["Gradient clipping", "5.0"],
            ["Checkpoint selection", "Validation loss"],
        ],
        [2.6, 3.4],
        "Table 4.3: Training hyperparameters shared by all variants.",
        font_size=10,
    )
    body_rich(
        [
            ("Evaluation protocol. ", True),
            ("BLEU-1–4 is computed with SacreBLEU and the 13a tokenizer; BLEU-k is the cumulative score over 1…k grams. CIDEr-D is implemented from scratch with global document frequencies, TF-IDF weighting and the standard length penalty. CHAIR follows the protocol of Rohrbach et al.: caption words are mapped to the 80 COCO classes with a curated synonym table, and a mention is hallucinated when the class is absent from the image's instance annotations. CHAIRs is the number of hallucinated unique mentions divided by all unique object mentions; CHAIRi is the fraction of captions with at least one hallucinated mention. Confidence intervals use percentile bootstrap over images, and paired comparisons resample the same image indices for both models. A McNemar test is used for the paired binary outcome “any hallucination”.", False),
        ]
    )
    body_rich(
        [
            ("Implementation and reproducibility. ", True),
            ("The pipeline is implemented in Python with PyTorch and torchvision. Feature extraction and detection run on Apple Metal Performance Shaders with a 55% memory cap; LSTM training runs on CPU. The repository contains scripts for data preparation, feature extraction, detector labelling, training, evaluation, figure generation and report building. Unit tests cover the CHAIR matcher, BLEU/CIDEr computation, all three model variants and every visualisation function. The experiment is designed to fit comfortably within 16 GB of unified memory.", False),
        ]
    )
    body_rich(
        [
            ("Design rationale. ", True),
            ("The three variants form a deliberately simple ladder. The global model establishes what can be achieved with one image vector. Attention adds spatial access without adding object knowledge. The object-aware model adds semantic evidence without changing the visual encoder. Holding the encoder and decoder fixed makes the comparison interpretable: any change in CHAIR can be attributed to the additional information, not to a larger backbone or a different training schedule. The same reasoning motivates the decision to select checkpoints by validation loss rather than validation BLEU, since BLEU-selected checkpoints can collapse to generic captions.", False),
        ]
    )
    body_rich(
        [
            ("Threats to validity. ", True),
            ("The main internal threat is detector noise: a false positive can teach the captioner to mention an absent object, while a missed detection can remove useful evidence. The main external threat is scale: a 4,000-image training set cannot represent the diversity of COCO train2017, so absolute scores are lower than published systems. These threats affect the size of the effects but not the logic of the controlled comparison, and they are revisited in the discussion of results.", False),
        ]
    )


def chapter_5():
    chapter_heading(["CHAPTER 5", "RESULTS AND DISCUSSION"])
    body(
        "The three models are evaluated on the same 500 held-out test images. Each model generates one greedy caption per image, and the caption is compared with the five COCO references and with the image's instance annotations. Table 5.1 reports caption quality; Table 5.2 reports grounding. The models are identical in architecture, optimiser and training budget except for the information supplied to the decoder."
    )
    add_table(
        ["Model", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr-D"],
        [
            ["Global", "56.84", "35.74", "20.65", "11.94", "0.485"],
            ["Attention", "59.75", "40.59", "26.09", "16.32", "0.704"],
            ["Object-aware", "61.96", "43.35", "28.44", "18.03", "0.815"],
        ],
        [1.5, 0.9, 0.9, 0.9, 0.9, 1.0],
        "Table 5.1: Caption-quality results on the 500-image test split. BLEU values are percentages; CIDEr-D is on a 0–10 scale.",
        font_size=10,
        align_center=True,
    )
    add_table(
        ["Model", "CHAIRs (%)", "CHAIRi (%)", "Object P (%)", "Object R (%)", "Object F1 (%)"],
        [
            ["Global", "49.32", "62.00", "50.68", "30.04", "37.72"],
            ["Attention", "34.01", "40.20", "65.99", "35.68", "46.31"],
            ["Object-aware", "23.64", "27.60", "76.36", "40.00", "52.50"],
        ],
        [1.5, 1.0, 1.0, 1.05, 1.05, 1.05],
        "Table 5.2: Object-hallucination and grounding results. Lower CHAIR values are better.",
        font_size=10,
        align_center=True,
    )
    body(
        "Caption quality improves monotonically from the global baseline to attention to object-aware fusion. BLEU-4 rises from 11.94 to 16.32 to 18.03, and CIDEr-D rises from 0.485 to 0.704 to 0.815. The improvement from attention is larger than the improvement from object labels, which is expected because attention changes the decoder's access to spatial evidence at every step, whereas object labels add a relatively low-dimensional auxiliary signal. The global baseline is clearly under-conditioned: it must compress the entire image into one vector before generating the first word."
    )
    body(
        "Grounding improves in the same order. CHAIRs falls from 49.32% to 34.01% to 23.64%, and CHAIRi falls from 62.00% to 40.20% to 27.60%. Object precision rises from 50.68% to 76.36%, and object F1 rises from 37.72% to 52.50%. The object-aware model therefore reduces both the proportion of hallucinated mentions and the proportion of images containing at least one hallucinated object. The result is consistent with the hypothesis that explicit object evidence helps the decoder avoid inventing objects, even when the evidence comes from a noisy detector."
    )
    add_figure(
        "model_comparison.png",
        6.2,
        "Figure 5.1: BLEU-4 and CHAIR hallucination rates for the three models. The object-aware model has both the highest BLEU-4 and the lowest hallucination rate.",
    )
    body(
        "The magnitude of the effect is substantial. The attention model removes about 0.30 hallucinated objects per image relative to the global baseline, and the object-aware model removes a further 0.16 per image relative to attention. The corresponding reductions in the fraction of images with any hallucination are 21.8 and 12.6 percentage points. These are average effects on 500 images; the qualitative examples later in this chapter show that individual captions can still fail."
    )
    section_heading("5.1 Paired Comparisons")
    body(
        "Because all models are evaluated on the same images, paired tests are more informative than independent confidence intervals. Table 5.3 reports the mean paired differences, bootstrap confidence intervals and McNemar p-values. Negative BLEU differences indicate that the second model is better; positive hallucination differences indicate that the first model hallucinates more."
    )
    add_table(
        ["Comparison", "Δ objects", "Δ rate (pp)", "Δ BLEU-4", "p"],
        [
            ["Global − Attention", "+0.30 [0.23, 0.37]", "+21.8 [17.0, 26.8]", "−3.27 [−4.18, −2.41]", "< 0.001"],
            ["Global − Object-aware", "+0.46 [0.39, 0.53]", "+34.4 [29.6, 39.0]", "−4.80 [−5.71, −3.85]", "< 0.001"],
            ["Attention − Object-aware", "+0.16 [0.11, 0.22]", "+12.6 [8.2, 17.0]", "−1.53 [−2.47, −0.60]", "< 0.001"],
        ],
        [1.65, 1.35, 1.15, 1.35, 0.7],
        "Table 5.3: Paired comparisons on the test split. Brackets show 95% bootstrap confidence intervals. McNemar tests for “any hallucination” are significant at p < 0.001 for all pairs.",
        font_size=9.5,
        align_center=True,
    )
    body(
        "Every pairwise difference is significant. Attention reduces hallucinated objects and improves BLEU relative to the global baseline. Object labels further reduce hallucination and improve BLEU relative to attention. The McNemar tests confirm that the change in the fraction of images with any hallucination is not due to chance: for the attention–object comparison, 100 images hallucinate only under attention, while 39 hallucinate only under the object-aware model. The direction of the effect is therefore consistent across images, not driven by a small subset."
    )
    section_heading("5.2 Does Better BLEU Mean Fewer Hallucinations?")
    body(
        "The model-level ordering suggests that better BLEU and better grounding can coincide. To examine the relationship more directly, Figure 5.2 plots sentence BLEU-4 against the number of hallucinated objects for every test caption, and Table 5.4 reports Spearman correlations. A negative correlation means that captions with higher BLEU tend to mention fewer absent objects."
    )
    add_figure(
        "bleu_vs_hallucination.png",
        5.9,
        "Figure 5.2: Sentence BLEU-4 versus hallucinated objects per caption for all 1,500 test captions. The black line shows the binned mean with a 95% bootstrap interval.",
    )
    add_table(
        ["Set", "N", "ρ (BLEU-4)", "p", "ρ (BLEU-2)", "p", "Halluc. rate (%)"],
        [
            ["Global", "500", "−0.33", "< 0.001", "−0.31", "< 0.001", "62.00"],
            ["Attention", "500", "−0.28", "< 0.001", "−0.25", "< 0.001", "40.20"],
            ["Object-aware", "500", "−0.25", "< 0.001", "−0.26", "< 0.001", "27.60"],
            ["Pooled", "1,500", "−0.33", "< 0.001", "−0.31", "< 0.001", "43.27"],
        ],
        [1.25, 0.55, 0.95, 0.7, 0.95, 0.7, 1.0],
        "Table 5.4: Spearman correlations between sentence BLEU and hallucination. The pooled p-value is descriptive; an image-cluster bootstrap gives ρ = −0.327 (95% CI −0.386 to −0.268).",
        font_size=9.5,
        align_center=True,
    )
    body(
        "The correlations are negative and statistically significant, but only moderate. At the caption level, higher BLEU is associated with fewer hallucinated objects, which is intuitive: captions that overlap the references more also tend to mention more of the right content. The pooled correlation is −0.327, and an image-cluster bootstrap that resamples the 500 images preserves the result (95% CI −0.386 to −0.268). The association is not an artefact of treating the three captions per image as independent observations."
    )
    body(
        "A correlation alone does not show that BLEU is sufficient. Table 5.5 therefore compares the models inside narrow BLEU bins. If BLEU fully explained hallucination, the three rows within a bin would be similar. In the 10–20 BLEU bin, however, the hallucination rates are 66.1%, 43.1% and 32.5% for global, attention and object-aware models, despite nearly identical mean BLEU. The object-aware model is consistently better in the low and middle bins, while the differences shrink in the highest bin, where all models produce conservative, reference-like captions."
    )
    add_table(
        ["BLEU-4 bin", "Model", "N", "Mean BLEU-4", "Halluc. rate (%)"],
        [
            ["0–10", "Global", "127", "7.0", "77.2"],
            ["0–10", "Attention", "108", "7.4", "57.4"],
            ["0–10", "Object-aware", "80", "7.2", "43.8"],
            ["10–20", "Global", "251", "13.4", "66.1"],
            ["10–20", "Attention", "209", "14.2", "43.1"],
            ["10–20", "Object-aware", "206", "14.2", "32.5"],
            ["20–30", "Global", "76", "24.4", "44.7"],
            ["20–30", "Attention", "105", "24.5", "30.5"],
            ["20–30", "Object-aware", "127", "24.7", "17.3"],
            ["30–40", "Global", "26", "34.7", "34.6"],
            ["30–40", "Attention", "45", "34.1", "28.9"],
            ["30–40", "Object-aware", "38", "35.0", "21.1"],
            ["40+", "Global", "20", "54.4", "15.0"],
            ["40+", "Attention", "33", "56.3", "12.1"],
            ["40+", "Object-aware", "49", "51.9", "12.2"],
        ],
        [0.9, 1.3, 0.6, 1.0, 1.2],
        "Table 5.5: Hallucination rate by sentence-BLEU bin. Models with nearly identical BLEU can differ substantially in grounding.",
        font_size=9.5,
        align_center=True,
    )
    body(
        "This is the strongest evidence against using BLEU as a grounding metric. BLEU captures part of caption quality, but it does not identify which model is grounded. A model can improve its n-gram overlap by producing safe, common phrases without becoming more faithful to the image. Conversely, a conservative caption may avoid hallucination while scoring lower on BLEU. Reporting CHAIR next to BLEU gives a more complete picture than either metric alone."
    )
    section_heading("5.3 Attention and Error Analysis", page_break=False)
    body(
        "Figure 5.3 shows word-level attention maps for four held-out images. Each column after the image visualises the attention distribution when the highlighted noun was generated. Localised attention on the named object is necessary but not sufficient for grounding: the model can look at the right region and still choose the wrong word."
    )
    add_figure(
        "attention_maps.png",
        4.5,
        "Figure 5.3: Word-level attention maps for the attention model. The highlighted words are object mentions selected from the generated captions.",
    )
    body(
        "The attention maps are often plausible. For the stop-sign image, the maps for “stop” and “sign” concentrate on the sign; for the bus image, the map for “bus” covers the bus; for the surfer image, the map for “man” is centred on the person. The teddy-bear example is more diffuse, and the model describes a bed that is not annotated, which illustrates the limitation of both attention and CHAIR: the model can attend to a plausible region while still generating a contextually common object that is absent."
    )
    add_figure(
        "qualitative_examples.png",
        5.8,
        "Figure 5.4: Qualitative comparison. Hallucinated object words are shown in red. The examples were selected to show disagreement between models, not to represent the average case.",
    )
    body(
        "The qualitative grid shows the same pattern as the aggregate metrics. The global model frequently invents a person, a table or a food item. The attention model reduces these errors but still adds objects that are plausible for the scene. The object-aware model is generally more conservative, although it can be misled by detector false positives. The examples also show that a caption can be fluent and partly correct while containing a single hallucinated object, which is precisely the failure that CHAIR is designed to expose."
    )
    body(
        "Two error patterns are worth noting. First, the models sometimes mention an object that is present but not annotated, such as a bed in the teddy-bear image; CHAIR counts this as a hallucination because the ground truth is incomplete. Second, the models sometimes use an ambiguous word, such as “glass” or “bat”, which the synonym table maps to a COCO class. These cases are a small fraction of the test set, but they illustrate why CHAIR should be interpreted alongside the annotations and the qualitative examples rather than as an absolute measure of truth."
    )
    section_heading("5.4 Discussion")
    body(
        "The experiment separates three effects that are often conflated. First, attention is not automatically grounding. It gives the decoder a spatial interface and improves both BLEU and CHAIR, but the training signal still rewards word likelihood. A model can learn useful attention maps and still over-generate familiar objects. Second, object labels are a strong but noisy prior. Detector labels make object vocabulary available at every step and reduce hallucination substantially, but false positives and missed detections propagate into the caption. Third, BLEU is a partial proxy. Better captions usually mention more of the right content, so BLEU and hallucination are negatively correlated, but the relationship is weak enough that equal BLEU does not imply equal grounding."
    )
    body(
        "The practical implication is that captioning evaluations should report a grounding metric next to BLEU or CIDEr. For applications where object correctness matters, a detector-label fusion layer is a relatively cheap intervention that can be added to an existing decoder. It does not require a large pretrained caption model or fine-tuning of the visual encoder, and it can be trained on a small dataset."
    )
    body(
        "The evaluation itself has limitations that are worth stating. CHAIR depends on COCO instance annotations, which are incomplete for small or unusual objects, and on a synonym table that can map an ambiguous word to the wrong class. The detector labels are noisy, the captions are decoded greedily, and the experiment uses one training run per model. These choices are appropriate for a reproducible laptop-scale study, but they mean that the absolute hallucination rates should be read as relative evidence rather than as universal estimates. A larger study with multiple seeds, beam search and human verification would provide a stronger estimate of the true error rate."
    )
    body(
        "Evaluation practice should therefore use a fixed panel of complementary measurements: a corpus-level quality metric, a caption-level quality metric, a grounding metric and per-image examples. Reporting the panel together makes it harder to mistake a higher BLEU score for a more truthful caption, and it gives readers the information needed to judge whether a model is suitable for their application."
    )
    body(
        "The results also caution against selecting models by BLEU alone. In this run, the BLEU leader was also the least hallucinating model, but the correlation was only moderate and matched-BLEU differences were large. A model slightly worse on BLEU but better on CHAIR may be preferable in a grounding-sensitive application."
    )


def conclusion():
    chapter_heading(["CONCLUSION"])
    body(
        "This project built and evaluated a controlled CNN–LSTM captioning benchmark for measuring object hallucination. The three models used the same visual features, decoder and training budget: a global baseline, a Bahdanau attention model and an object-aware model with detected labels. On 500 held-out COCO images, the object-aware model achieved the best BLEU-4 (18.03) and the lowest CHAIRs (23.64%), followed by attention (16.32, 34.01%) and the global baseline (11.94, 49.32%). All pairwise differences were significant."
    )
    body(
        "The central finding is that BLEU and hallucination are related but not interchangeable. Sentence BLEU-4 and hallucinated-object count had a moderate negative Spearman correlation (ρ = −0.327), and the lowest-BLEU quartile had a much higher hallucination rate than the highest-BLEU quartile. Yet models with nearly identical BLEU in the 10–20 bin differed by more than 30 percentage points in hallucination rate. BLEU captures part of caption quality, but it does not identify which model is grounded."
    )
    body(
        "The project makes three contributions. It provides a reproducible, laptop-scale pipeline; it quantifies the effect of attention and noisy object labels under a controlled protocol; and it demonstrates an evaluation pattern that combines BLEU, CIDEr-D and CHAIR with paired statistics and attention visualisation. The code and artifacts are included so that the experiment can be repeated and extended."
    )
    body(
        "The main limitations are the small training set, the re-split of COCO validation images, incomplete annotations, detector noise and a single run. Future work should use the full COCO split, beam search and human verification, and test whether the BLEU–CHAIR relationship holds across architectures and datasets."
    )


def references():
    chapter_heading(["REFERENCES"])
    refs = [
        "Vinyals, O., Toshev, A., Bengio, S., and Erhan, D. (2015). Show and Tell: A Neural Image Caption Generator. CVPR. arXiv:1411.4555.",
        "Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., and Bengio, Y. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML. arXiv:1502.03044.",
        "Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. (2018). Object Hallucination in Image Captioning. EMNLP. arXiv:1809.02156.",
        "Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., and Zhang, L. (2018). Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering. CVPR. arXiv:1707.07998.",
        "Vedantam, R., Zitnick, C. L., and Parikh, D. (2015). CIDEr: Consensus-based Image Description Evaluation. CVPR. arXiv:1411.5726.",
        "Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL.",
        "Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. ECCV. arXiv:1405.0312.",
        "Ren, S., He, K., Girshick, R., and Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS. arXiv:1506.01497.",
        "He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. arXiv:1512.03385.",
        "You, Q., Jin, H., Wang, Z., Fang, C., and Luo, J. (2016). Image Captioning with Semantic Attention. CVPR.",
        "Lu, J., Xiong, C., Parikh, D., and Socher, R. (2017). Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning. CVPR.",
        "Hendricks, L. A., Burns, K., Saenko, K., Darrell, T., and Rohrbach, A. (2018). Women Also Snowboard: Overcoming Bias in Captioning Models. ECCV.",
    ]
    for index, text in enumerate(refs, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.1
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(f"{index}) {text}")
        set_run_font(run, 11.5)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build(out_path: Path, page_map: dict) -> None:
    global doc
    doc = Document()
    configure_section(doc.sections[0])
    # Remove the default empty paragraph.
    if doc.paragraphs:
        p = doc.paragraphs[0]._element
        p.getparent().remove(p)

    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(BODY_PT)
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.line_spacing = 1.5

    # Section 1: cover, certificate, acknowledgement (no page numbers).
    cover_page()
    certificate_page()
    acknowledgement_page()

    # Section 2: front matter with roman page numbers.
    front = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(front)
    add_footer(front, "upperRoman")
    abstract_page()
    toc_page(page_map)
    list_page(page_map)
    abbreviations_page()

    # Section 3: body with decimal page numbers starting at 1.
    body_section = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(body_section)
    add_footer(body_section, "decimal")
    chapter_1()
    chapter_2()
    chapter_3()
    chapter_4()
    chapter_5()
    conclusion()
    references()

    # Metadata.
    doc.core_properties.title = TITLE
    doc.core_properties.author = "<Student Name>"
    doc.core_properties.subject = "U21CS503 Mobile Application Development Report"
    doc.core_properties.comments = "Report generated in the KPR / MAD report format."

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--page-map", default=str(PAGE_MAP_PATH))
    args = parser.parse_args()
    page_map = {}
    path = Path(args.page_map)
    if path.exists():
        page_map = json.loads(path.read_text(encoding="utf-8"))
    out = Path(args.out)
    build(out, page_map)
    print(out)


if __name__ == "__main__":
    main()
