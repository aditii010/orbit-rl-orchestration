import html as _html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

import build_html_lists as L
from content import APPENDIX_A, APPENDIX_B, B, REFS
from front import front_blocks

W = 9026
PAGES = json.load(open("report/pages.json")) if os.path.exists("report/pages.json") else {}
OUT = []
add = OUT.append


def esc(s):
    s = _html.unescape(s)
    o = []
    for ch in s:
        n = ord(ch)
        if ch in "\\{}":
            o.append("\\" + ch)
        elif n < 128:
            o.append(ch)
        elif n < 256:
            o.append("\\'%02x" % n)
        elif n <= 0x3FF:
            o.append("\\u%d?" % n)
        else:
            raise ValueError("unsupported character %r" % ch)
    return "".join(o)


def inl(s):
    s = esc(s)
    for tag, on in (("b", "{\\b "), ("i", "{\\i "), ("hl", "{\\highlight7 ")):
        s = s.replace("<%s>" % tag, on).replace("</%s>" % tag, "}")
    return s


def p(text, align="qj", sa=80, size=24):
    add("\\pard\\plain\\%s\\sl276\\slmult1\\sa%d\\fs%d %s\\par\n" % (align, sa, size, inl(text)))


def heading(text, level, newpage=False, centered=False):
    size, sb = {1: (28, 220), 2: (24, 160)}[level]
    add("\\pard%s\\s%d\\outlinelevel%d\\keepn\\sb%d\\sa80\\%s\\plain\\b\\fs%d %s\\par\n" % ("\\pagebb" if newpage else "", level, level - 1, sb, "qc" if centered else "ql", size, inl(text)))


def plain_title(text, newpage=False, sb=0):
    add("\\pard%s\\keepn\\sb%d\\sa100\\qc\\plain\\b\\fs28 %s\\par\n" % ("\\pagebb" if newpage else "", sb, inl(text)))


def line(text, size=24, align="qc", bold=False, italic=False, sb=0, sa=60, color=None):
    c = "\\cf%d" % color if color else ""
    add("\\pard\\plain\\%s\\sb%d\\sa%d%s%s%s\\fs%d %s\\par\n" % (align, sb, sa, "\\b" if bold else "", "\\i" if italic else "", c, size, inl(text)))


def spacer(pts):
    add("\\pard\\plain\\sa0\\fs%d \\par\n" % max(2, int(pts * 2)))


def bullet(text):
    add("\\pard\\plain\\ls1\\ilvl0\\fi-360\\li720\\qj\\sl276\\slmult1\\sa40\\fs24 %s\\par\n" % inl(text))


def row(widths, cells, header=False, aligns=None, size=20):
    x, d = 0, []
    for w in widths:
        x += w
        d.append("%s\\cellx%d" % ("\\clcbpat3" if header else "", x))
    o = "\\trowd\\trgaph70%s\n%s\n" % ("\\trhdr" if header else "", "".join(d))
    for i, c in enumerate(cells):
        al = aligns[i] if aligns else "l"
        t = inl(c)
        if header:
            t = "{\\b " + t + "}"
        o += "\\pard\\intbl%s\\fs%d %s\\cell" % ("\\qc" if (header or al == "c") else "", size, t)
    add(o + "\\row\n")


def table(head, rows, widths, aligns, size=20):
    row(widths, head, header=True, size=size)
    for r in rows:
        row(widths, r, aligns=aligns, size=size)
    spacer(5)


def caption(text, above):
    lab, rest = text.split(":", 1)
    add("\\pard\\plain\\qc\\sb%d\\sa%d%s\\fs20 {\\b %s:}%s\\par\n" % (100 if above else 40, 60 if above else 100, "\\keepn" if above else "", esc(lab), inl(rest)))


def placeholder(fname, width_in, text):
    im = Image.open("report/img/" + fname)
    h_in = width_in * im.height / im.width
    lab = text.split(":")[0]
    n = max(3, int(h_in * 72 / 11.0) - 1)
    note = "[ Paste %s here: %s, about %.1f in wide ]" % (lab, fname.split(".")[0].replace("_", " "), width_in)
    add("\\trowd\\trgaph70\\clvertalc\\cellx%d\n\\pard\\intbl\\qc\\i\\cf10\\fs18 %s%s\\cell\\row\n" % (W, esc(note), "\\par" * n))
    caption(text, above=False)


def toc_line(text, key, indent=0, bold=False, size=20):
    add("\\pard\\plain\\tqr\\tldot\\tx%d\\li%d\\sa30%s\\fs%d %s\\tab %s\\par\n" % (W, indent, "\\b" if bold else "", size, inl(text), PAGES.get(key, "00")))


for b in front_blocks():
    k = b[0]
    if k == "center":
        _, t, sz, bold, ital, sa = b[:6]
        col = 2 if (len(b) > 6 and b[6]) else None
        line(t, sz * 2, "qc", bold, ital, sb=(b[7] * 20 if len(b) > 7 else 0), sa=sa * 20, color=col)
    elif k == "cover_table":
        wd = [W // 2, W - W // 2]
        row(wd, b[1], header=True, size=22)
        for r in b[2]:
            row(wd, r, aligns=["c", "c"], size=22)
        spacer(8)
    elif k == "h1c":
        heading(b[1], 1, newpage=b[2], centered=True)
    elif k == "p":
        p(b[1])
    elif k == "right":
        line(b[1], 24, "qr", bold=b[2], sa=40)
    elif k == "left":
        line(b[1], 24, "ql", sa=40)
    elif k == "spacer":
        spacer(b[1])

plain_title("TABLE OF CONTENTS", newpage=True)
add("\\pard\\plain\\qc\\sa100\\fs22 {\\highlight7 [ In Google Docs: click here, then Insert > Table of contents. It fills itself with the headings and page numbers. Delete this line afterwards. ]}\\par\n")
def plain_item(text, size=19, indent=0):
    add("\\pard\\plain\\li%d\\sa30\\fs%d %s\\par\\n" % (indent, size, inl(text)))
plain_title("LIST OF TABLES", sb=200)
for b in B:
    if b[0] == "table":
        plain_item(b[1])
plain_item("Table B.1: Predictions written before the runs, and what happened")
plain_title("LIST OF FIGURES", sb=200)
for b in B:
    if b[0] == "fig":
        plain_item(b[2])

for b in B:
    k = b[0]
    if k == "h1":
        heading(b[1], 1, newpage=(b[1].startswith("1.")))
    elif k == "h2":
        heading(b[1], 2)
    elif k == "p":
        p(b[1])
    elif k == "bullets":
        for x in b[1]:
            bullet(x)
    elif k == "fig":
        placeholder(b[1], b[3], b[2])
    elif k == "table":
        _, cap, head, rows, widths, al = b
        caption(cap, above=True)
        table(head, rows, [int(w * 1440) for w in widths], al)

heading("REFERENCES", 1)
for i, t in enumerate(REFS, 1):
    add("\\pard\\plain\\qj\\sa50\\li600\\fi-600\\fs21 [%d]\\tab %s\\par\n" % (i, inl(t)))
heading("APPENDIX A: CODE ORGANISATION AND REPRODUCTION", 1)
p(APPENDIX_A)
heading("APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES", 1)
caption("Table B.1: Predictions written before the runs, and what happened", above=True)
table(["Experiment", "Prediction or rule", "Outcome"], APPENDIX_B, [2400, 2900, 3726], ["l", "l", "l"])

fonts = "{\\fonttbl{\\f0\\froman\\fcharset0 Times New Roman;}}"
cols = ("{\\colortbl;\\red0\\green0\\blue0;\\red31\\green56\\blue100;\\red217\\green226\\blue243;\\red71\\green86\\blue90;\\red173\\green182\\blue206;"
        "\\red165\\green96\\blue31;\\red255\\green255\\blue0;\\red124\\green129\\blue32;\\red234\\green240\\blue250;\\red120\\green120\\blue120;}")
sty = ("{\\stylesheet{\\s0\\f0\\fs24 Normal;}{\\s1\\f0\\fs28\\b\\keepn\\outlinelevel0 heading 1;}{\\s2\\f0\\fs24\\b\\keepn\\outlinelevel1 heading 2;}}")
lst = ("{\\*\\listtable{\\list\\listtemplateid1{\\listlevel\\levelnfc23\\leveljc0\\levelstartat1\\levelfollow0\\levelspace0\\levelindent0"
       "{\\leveltext\\'01\\'95;}{\\levelnumbers;}\\f0\\fi-360\\li720}{\\listname bullets;}\\listid1}}\n{\\*\\listoverridetable{\\listoverride\\listid1\\listoverridecount0\\ls1}}")
head = "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang2057\n%s\n%s\n%s\n%s\n\\paperw11906\\paperh16838\\margl1440\\margr1440\\margt1440\\margb1440\\widowctrl\n" % (fonts, cols, sty, lst)
rtf = head + "{\\footer\\pard\\plain\\qc\\fs20 \\chpgn\\par}\n\\sectd\n" + "".join(OUT) + "}\n"
assert rtf.count("{") - rtf.count("\\{") == rtf.count("}") - rtf.count("\\}"), "unbalanced braces"
assert all(ord(c) < 128 for c in rtf), "non-ascii"
open("report/ORBIT_Report.rtf", "w").write(rtf)
print("RTF bytes:", len(rtf), "| lines:", rtf.count("\n"))
