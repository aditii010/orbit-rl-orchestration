import html as _html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

import build_html_lists as L
from content import TITLE, APPENDIX_A, APPENDIX_B, B, REFS
from front import front_blocks

PAGES = json.load(open("report/pages.json"))
TNR = "Times New Roman"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.54)
TEXT_W = 6.27


def set_font(style, size, bold=None, italic=None, color=RGBColor(0, 0, 0)):
    style.font.name = TNR
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), TNR)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(k)) is not None:
            del rf.attrib[qn(k)]


st = doc.styles
set_font(st["Normal"], 12)
st["Normal"].paragraph_format.line_spacing = 1.15
st["Normal"].paragraph_format.space_after = Pt(4)
set_font(st["Heading 1"], 14, bold=True)
st["Heading 1"].paragraph_format.space_before = Pt(11); st["Heading 1"].paragraph_format.space_after = Pt(4); st["Heading 1"].paragraph_format.keep_with_next = True
set_font(st["Heading 2"], 12, bold=True)
st["Heading 2"].paragraph_format.space_before = Pt(8); st["Heading 2"].paragraph_format.space_after = Pt(3); st["Heading 2"].paragraph_format.keep_with_next = True
set_font(st["Caption"], 10, bold=False, italic=False)
set_font(st["List Bullet"], 12)
st["List Bullet"].paragraph_format.space_after = Pt(2)


def runs(par, text, size=None, bold=None, italic=None, color=None):
    text = _html.unescape(text)
    state = {"b": bool(bold), "i": bool(italic), "hl": False}
    for tok in re.split(r"(</?(?:b|i|hl)>)", text):
        m = re.fullmatch(r"<(/?)(b|i|hl)>", tok)
        if m:
            state[m.group(2)] = not m.group(1)
            continue
        if not tok:
            continue
        r = par.add_run(tok)
        r.bold, r.italic = state["b"] or None, state["i"] or None
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        if state["hl"]:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par


def para(text="", align=None, size=None, bold=None, italic=None, sa=None, sb=None, color=None, style=None, keep=False):
    p = doc.add_paragraph(style=style)
    if align:
        p.alignment = align
    if sa is not None:
        p.paragraph_format.space_after = Pt(sa)
    if sb is not None:
        p.paragraph_format.space_before = Pt(sb)
    if keep:
        p.paragraph_format.keep_with_next = True
    runs(p, text, size, bold, italic, color)
    return p


J, C, R_ = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def heading(text, level, newpage=False, centered=False):
    p = doc.add_heading(level=level)
    runs(p, text)
    if centered:
        p.alignment = C
    if newpage:
        p.paragraph_format.page_break_before = True
    return p


def cell_text(cell, text, size, bold=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
    if align:
        p.alignment = align
    runs(p, text, size, bold)


def make_table(head, rows, widths, aligns, size=10, header_fill="D9E2F3"):
    t = doc.add_table(rows=1 + len(rows), cols=len(head))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": C}
    for j, h in enumerate(head):
        cell_text(t.rows[0].cells[j], h, size, True, C); shade(t.rows[0].cells[j], header_fill)
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r):
            cell_text(t.rows[i].cells[j], v, size, False, amap[aligns[j]])
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for j, w in enumerate(widths):
            row.cells[j].width = Inches(w)
    hdr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); hdr.append(th)
    return t


def tab_line(text, page, size=10, bold=False, indent=0.0, sa=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(sa); p.paragraph_format.line_spacing = 1.0; p.paragraph_format.left_indent = Inches(indent)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    runs(p, text + "\t" + str(page), size, bold)
    return p


def fld(par, kind, first=False):
    r = OxmlElement("w:r"); c = OxmlElement("w:fldChar"); c.set(qn("w:fldCharType"), kind); r.append(c)
    if first:
        par._p.insert(1 if par._p.pPr is not None else 0, r)
    else:
        par._p.append(r)
    return r


def instr(par, text, pos):
    r = OxmlElement("w:r"); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; r.append(it)
    par._p.insert(pos, r)


def toc_control(entries):
    paras = [tab_line(t, pg, size=10, bold=(lv == 1), indent=0.0 if lv == 1 else 0.25) for lv, t, pg in entries]
    sdt = parse_xml('<w:sdt %s><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/><w:docPartUnique/></w:docPartObj></w:sdtPr><w:sdtContent/></w:sdt>' % nsdecls("w"))
    paras[0]._p.addprevious(sdt)
    content = sdt.find(qn("w:sdtContent"))
    for p in paras:
        content.append(p._p)
    base = 1 if paras[0]._p.pPr is not None else 0
    r1 = OxmlElement("w:r"); c1 = OxmlElement("w:fldChar"); c1.set(qn("w:fldCharType"), "begin"); r1.append(c1)
    r2 = OxmlElement("w:r"); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = ' TOC \\o "1-2" \\h \\z \\u '; r2.append(it)
    r3 = OxmlElement("w:r"); c3 = OxmlElement("w:fldChar"); c3.set(qn("w:fldCharType"), "separate"); r3.append(c3)
    for k, r in enumerate((r1, r2, r3)):
        paras[0]._p.insert(base + k, r)
    r4 = OxmlElement("w:r"); c4 = OxmlElement("w:fldChar"); c4.set(qn("w:fldCharType"), "end"); r4.append(c4)
    paras[-1]._p.append(r4)


fp = sec.footer.paragraphs[0]
fp.alignment = C
for kind, txt in (("begin", None), (None, " PAGE "), ("end", None)):
    r = fp.add_run()
    if kind:
        c = OxmlElement("w:fldChar"); c.set(qn("w:fldCharType"), kind); r._r.append(c)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)
    r.font.size = Pt(10)

for b in front_blocks():
    k = b[0]
    if k == "center":
        _, t, sz, bold, ital, sa = b[:6]
        col = b[6] if len(b) > 6 else None
        sb = b[7] if len(b) > 7 else 0
        para(t, C, sz, bold, ital, sa=sa, sb=sb, color=col)
    elif k == "cover_table":
        t = doc.add_table(rows=1 + len(b[2]), cols=2); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(b[1]):
            cell_text(t.rows[0].cells[j], h, 11, True, C); shade(t.rows[0].cells[j], "D9E2F3")
        for i, r in enumerate(b[2], 1):
            for j, v in enumerate(r):
                cell_text(t.rows[i].cells[j], v, 11, False, C)
        para("", sa=6)
    elif k == "h1c":
        heading(b[1], 1, newpage=b[2], centered=True)
    elif k == "p":
        para(b[1], J)
    elif k == "right":
        para(b[1], R_, bold=b[2] or None, sa=2)
    elif k == "left":
        para(b[1], sa=2)
    elif k == "spacer":
        para("", sa=0, sb=b[1] / 2)

heading("TABLE OF CONTENTS", 1, newpage=True, centered=True).style = st["Normal"]
h = doc.paragraphs[-1]; h.alignment = C
for r in h.runs:
    r.bold = True; r.font.size = Pt(14)
entries = [(1, t, PAGES["H:" + t]) for t in ("CERTIFICATE", "DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT")]
for b in B:
    if b[0] in ("h1", "h2"):
        entries.append((1 if b[0] == "h1" else 2, b[1], PAGES["H:" + b[1]]))
for t in ("REFERENCES", "APPENDIX A: CODE ORGANISATION AND REPRODUCTION", "APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES"):
    entries.append((1, t, PAGES["H:" + t]))
toc_control(entries)
for title, kind in (("LIST OF TABLES", "table"), ("LIST OF FIGURES", "fig")):
    p = para(title, C, 14, True, sb=10, sa=4, keep=True)
    for lab, text in [(b[1].split(":")[0], b[1]) if b[0] == "table" else (b[2].split(":")[0], b[2]) for b in B if b[0] == kind] + ([("Table B.1", "Table B.1: Predictions written before the runs, and what happened")] if kind == "table" else []):
        tab_line(text, PAGES["C:" + lab], size=9.5)

for b in B:
    k = b[0]
    if k == "h1":
        heading(b[1], 1, newpage=b[2])
    elif k == "h2":
        heading(b[1], 2)
    elif k == "p":
        para(b[1], J)
    elif k == "bullets":
        for x in b[1]:
            p = doc.add_paragraph(style="List Bullet"); p.alignment = J; runs(p, x)
    elif k == "fig":
        pp = doc.add_paragraph(); pp.alignment = C; pp.paragraph_format.keep_with_next = True; pp.paragraph_format.space_after = Pt(2)
        pp.add_run().add_picture("report/img/" + b[1], width=Inches(b[3]))
        lab, rest = b[2].split(":", 1)
        c = para("", C, 10, sa=6); runs(c, "<b>%s:</b>%s" % (lab, rest), 10)
    elif k == "table":
        _, cap, head, rows, widths, al = b
        lab, rest = cap.split(":", 1)
        c = para("", C, 10, sb=6, sa=3, keep=True); runs(c, "<b>%s:</b>%s" % (lab, rest), 10)
        make_table(head, rows, widths, al)
        para("", sa=2).paragraph_format.line_spacing = 0.6

heading("REFERENCES", 1)
for i, t in enumerate(REFS, 1):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.42); p.paragraph_format.first_line_indent = Inches(-0.42)
    p.paragraph_format.space_after = Pt(3); p.paragraph_format.line_spacing = 1.05; p.alignment = J
    runs(p, "[%d]\t%s" % (i, t), 10.5)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.42))
heading("APPENDIX A: CODE ORGANISATION AND REPRODUCTION", 1)
para(APPENDIX_A, J)
heading("APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES", 1)
c = para("", C, 10, sb=4, sa=3, keep=True); runs(c, "<b>Table B.1:</b> Predictions written before the runs, and what happened", 10)
make_table(["Experiment", "Prediction or rule", "Outcome"], APPENDIX_B, [1.7, 2.0, 2.57], ["l", "l", "l"])

cp = doc.core_properties
cp.title = TITLE; cp.author = "Aditi Sikarwar"; cp.subject = "UCS760 project report"; cp.keywords = "reinforcement learning, Double DQN, multi-agent LLM"
doc.save("report/ORBIT_Report.docx")
print("saved", os.path.getsize("report/ORBIT_Report.docx") // 1024, "KB")
