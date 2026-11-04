"""Tiny RTF writer for the ORBIT report. Google Docs imports RTF with real page breaks, headings (outline levels), lists,
shaded tables and exact column widths, which its HTML import does not. Plain ASCII output only (plus a few cp1252 escapes).
Findings that shaped this file (tested against the Drive converter): stylesheet formatting is ignored, so every paragraph carries its own
direct formatting; \\plain is needed to stop bold leaking between paragraphs; table borders cannot be removed; per-section footers are
merged into one footer; code points above U+03FF come out garbled."""
import json
import os
import re

W = 9026                      # text width in twips: A4 (11906) minus 2 x 1 inch margins
COL = dict(black=1, navy=2, light=3, slate=4, greyblue=5, copper=6, yellow=7, olive=8, paleblue=9, grey=10)
HEAD, BODY = [], []           # BODY = rtf pieces; HEAD = (level, text) registry for the contents list
CAPS = []                     # (kind, number, text) for the list of tables / figures
PAGES = {}                    # filled from report/toc_pages.json after the first build
_n = {"table": 0, "figure": 0}
_lists = {"numbered": 0}


def esc(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\\{}":
            out.append("\\" + ch)
        elif o < 128:
            out.append(ch)
        elif o < 256:
            out.append("\\'%02x" % o)
        elif o <= 0x3FF:
            out.append("\\u%d?" % o)
        else:
            raise ValueError("unsupported character %r in %r" % (ch, s[:60]))
    return "".join(out)


def inline(s):
    """Escape then apply tiny markup: <b> <i> <sub> <sup> <hl> (highlighted placeholder)."""
    s = esc(s)
    for tag, on in (("b", "{\\b "), ("i", "{\\i "), ("sub", "{\\sub "), ("sup", "{\\super "), ("hl", "{\\highlight%d " % COL["yellow"])):
        s = s.replace("<%s>" % tag, on).replace("</%s>" % tag, "}")
    return s


def add(piece):
    BODY.append(piece)


# ---------------------------------------------------------------- paragraphs
def p(text, indent=True, sa=100, align="qj", size=24):
    add("\\pard\\plain\\%s\\sl360\\slmult1\\sa%d%s\\fs%d %s\\par\n" % (align, sa, "\\fi720" if indent else "", size, inline(text)))


def heading(level, text, newpage=False, centered=False):
    size, sb = {1: (32, 240), 2: (26, 200), 3: (24, 160)}[level]
    HEAD.append((level, text))
    it = "\\i" if level == 3 else ""
    add("\\pard%s\\s%d\\outlinelevel%d\\keepn\\sb%d\\sa120%s\\plain\\b%s\\fs%d %s\\par\n"
        % ("\\pagebb" if newpage else "", level, level - 1, sb, "\\qc" if centered else "\\ql", it, size, inline(text)))


def front_title(text):
    """Heading 1 on a new page, centred (certificate, declaration, ...)."""
    heading(1, text, newpage=True, centered=True)


def bullet(text, sa=60):
    add("\\pard\\plain\\ls1\\ilvl0\\fi-360\\li720\\qj\\sl360\\slmult1\\sa%d\\fs24 %s\\par\n" % (sa, inline(text)))


def numbered(items, sa=60):
    _lists["numbered"] += 1
    ls = 1 + _lists["numbered"]
    for t in items:
        add("\\pard\\plain\\ls%d\\ilvl0\\fi-360\\li720\\qj\\sl360\\slmult1\\sa%d\\fs24 %s\\par\n" % (ls, sa, inline(t)))


def spacer(pts=12):
    add("\\pard\\plain\\sa0\\fs%d \\par\n" % (pts * 2))


def line(text, size=24, align="qc", bold=False, italic=False, sb=0, sa=60, color=None):
    c = "\\cf%d" % COL[color] if color else ""
    add("\\pard\\plain\\%s\\sb%d\\sa%d%s%s%s\\fs%d %s\\par\n" % (align, sb, sa, "\\b" if bold else "", "\\i" if italic else "", c, size, inline(text)))


def reference(text):
    add("\\pard\\plain\\qj\\sl300\\slmult1\\sa80\\li720\\fi-720\\fs22 %s\\par\n" % inline(text))


# ---------------------------------------------------------------- captions
class Cap:
    counters = {}


def _cap(kind, chapter, text, keep):
    key = (kind, chapter)
    Cap.counters[key] = Cap.counters.get(key, 0) + 1
    label = "%s %s.%d" % (kind.capitalize(), chapter, Cap.counters[key])
    CAPS.append((kind, label, text))
    add("\\pard\\plain\\qc\\sb120\\sa80%s\\b\\fs20 %s: %s\\par\n" % ("\\keepn" if keep else "", label, inline(text)))
    return label


def table_caption(chapter, text):
    return _cap("table", chapter, text, keep=True)


def figure_caption(chapter, text):
    return _cap("figure", chapter, text, keep=False)


# ---------------------------------------------------------------- tables
def _row(widths, cells, header=False, fills=None, aligns=None, size=20, bold_first=False, height=None):
    x, defs = 0, []
    for i, w in enumerate(widths):
        x += w
        f = (fills[i] if fills and fills[i] else (COL["light"] if header else 0))
        defs.append("%s\\cellx%d" % ("\\clcbpat%d" % f if f else "", x))
    out = ["\\trowd\\trgaph70%s%s\n" % ("\\trhdr" if header else "", ("\\trrh%d" % height) if height else "")]
    out.append("".join(defs) + "\n")
    for i, c in enumerate(cells):
        al = (aligns[i] if aligns else "ql")
        txt = inline(c)
        if header or (bold_first and i == 0):
            txt = "{\\b " + txt + "}"
        out.append("\\pard\\intbl%s\\fs%d %s\\cell" % ("" if al == "ql" else "\\" + al, size, txt))
    out.append("\\row\n")
    return "".join(out)


def table(widths, header, rows, aligns=None, size=20, bold_first=False):
    assert sum(widths) <= W + 5, sum(widths)
    add(_row(widths, header, header=True, aligns=aligns, size=size))
    for r in rows:
        add(_row(widths, r, aligns=aligns, size=size, bold_first=bold_first))
    spacer(6)


def shaded_row(widths, cells, fills, aligns=None, size=20, bold=True):
    add(_row(widths, cells, fills=fills, aligns=aligns, size=size, bold_first=bold))


# ---------------------------------------------------------------- charts built from shaded table cells
def bars(rows, vmax, label_w=3000, bar_w=4700, val_w=1300, size=20):
    """Horizontal bar chart. rows = (label, value, colour name, value text). Bar length is exact (twips)."""
    for label, v, colour, vtext in rows:
        L = max(30, int(bar_w * min(abs(v), vmax) / vmax))
        x1, x2, x3, x4 = label_w, label_w + L, label_w + bar_w, label_w + bar_w + val_w
        c = COL[colour]
        if L >= bar_w:
            defs = "\\cellx%d\\clcbpat%d\\cellx%d\\cellx%d" % (x1, c, x3, x4)
            cells = [(label, "ql", size), ("", "ql", 8), (vtext, "qr", size)]
        else:
            defs = "\\cellx%d\\clcbpat%d\\cellx%d\\cellx%d\\cellx%d" % (x1, c, x2, x3, x4)
            cells = [(label, "ql", size), ("", "ql", 8), ("", "ql", 8), (vtext, "qr", size)]
        out = "\\trowd\\trgaph50\\trrh-270%s\n" % defs
        out = out.replace("\\cellx%d\\clcbpat" % x1, "\\cellx%d\\clcbpat" % x1)
        for t, al, sz in cells:
            out += "\\pard\\intbl%s\\fs%d %s\\cell" % ("" if al == "ql" else "\\" + al, sz, inline(t))
        out += "\\row\n"
        add(out)
    spacer(6)


def vchart(values, ymax, rows=8, labels=None, colour="navy", col_w=560, label_w=700, size=16, ylabels=True):
    """Vertical bar chart: len(values) columns, `rows` levels. labels = x labels under the columns."""
    n = len(values)
    c = COL[colour]
    for r in range(rows, 0, -1):                       # top row = highest level
        level = ymax * r / rows
        x, defs = label_w, ["\\cellx%d" % label_w]
        for v in values:
            x += col_w
            defs.append("%s\\cellx%d" % ("\\clcbpat%d" % c if v >= level - ymax / rows / 2 + 1e-9 else "", x))
        ylab = ("%.2f" % level) if (ylabels and r % 2 == 0) else ""
        out = "\\trowd\\trgaph0\\trrh-250%s\n" % "".join(defs)
        out += "\\pard\\intbl\\qr\\fs%d %s\\cell" % (size, ylab)
        out += "".join("\\pard\\intbl\\fs4\\cell" for _ in values) + "\\row\n"
        add(out)
    x, defs = label_w, ["\\cellx%d" % label_w]
    for _ in values:
        x += col_w
        defs.append("\\cellx%d" % x)
    out = "\\trowd\\trgaph0%s\n\\pard\\intbl\\fs%d \\cell" % ("".join(defs), size)
    for lab in (labels or [""] * n):
        out += "\\pard\\intbl\\qc\\fs%d %s\\cell" % (size, lab)
    add(out + "\\row\n")
    spacer(6)


def flow(boxes, widths=None, fill="light", size=20):
    """One-row block diagram: boxes joined by arrow cells."""
    n = len(boxes)
    aw = 420
    bw = (W - aw * (n - 1)) // n
    x, defs, cells = 0, [], []
    for i, b in enumerate(boxes):
        x += bw
        defs.append("\\clcbpat%d\\cellx%d" % (COL[fill], x))
        cells.append((b, "qc"))
        if i < n - 1:
            x += aw
            defs.append("\\cellx%d" % x)
            cells.append(("->", "qc"))
    out = "\\trowd\\trgaph70%s\n" % "".join(defs)
    for t, al in cells:
        out += "\\pard\\plain\\intbl\\%s\\fs%d %s\\cell" % (al, size, inline(t))
    add(out + "\\row\n")
    spacer(6)


# ---------------------------------------------------------------- contents lists
def toc_line(text, key, indent=0, bold=False, size=22):
    pg = PAGES.get(key, "00")
    add("\\pard\\plain\\tqr\\tldot\\tx%d\\li%d\\sa40%s\\fs%d %s\\tab %s\\par\n" % (W, indent, "\\b" if bold else "", size, inline(text), pg))


# ---------------------------------------------------------------- document assembly
def document(footer=True):
    fonts = "{\\fonttbl{\\f0\\froman\\fcharset0 Times New Roman;}{\\f1\\fswiss\\fcharset0 Arial;}}"
    cols = ("{\\colortbl;\\red0\\green0\\blue0;\\red31\\green56\\blue100;\\red217\\green226\\blue243;\\red71\\green86\\blue90;"
            "\\red173\\green182\\blue206;\\red165\\green96\\blue31;\\red255\\green255\\blue0;\\red124\\green129\\blue32;\\red234\\green240\\blue250;\\red242\\green242\\blue242;}")
    sty = ("{\\stylesheet{\\s0\\f0\\fs24 Normal;}{\\s1\\f0\\fs32\\b\\keepn\\outlinelevel0 heading 1;}"
           "{\\s2\\f0\\fs26\\b\\keepn\\outlinelevel1 heading 2;}{\\s3\\f0\\fs24\\b\\i\\keepn\\outlinelevel2 heading 3;}}")
    lt = ["{\\list\\listtemplateid1{\\listlevel\\levelnfc23\\leveljc0\\levelstartat1\\levelfollow0\\levelspace0\\levelindent0"
          "{\\leveltext\\'01\\'95;}{\\levelnumbers;}\\f0\\fi-360\\li720}{\\listname bullets;}\\listid1}"]
    lo = ["{\\listoverride\\listid1\\listoverridecount0\\ls1}"]
    for k in range(1, _lists["numbered"] + 1):
        lt.append("{\\list\\listtemplateid%d{\\listlevel\\levelnfc0\\leveljc0\\levelstartat1\\levelfollow0\\levelspace0\\levelindent0"
                  "{\\leveltext\\'02\\'00.;}{\\levelnumbers\\'01;}\\f0\\fi-360\\li720}{\\listname n%d;}\\listid%d}" % (k + 1, k, k + 1))
        lo.append("{\\listoverride\\listid%d\\listoverridecount0\\ls%d}" % (k + 1, k + 1))
    head = ("{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang2057\n%s\n%s\n%s\n{\\*\\listtable%s}\n{\\*\\listoverridetable%s}\n"
            "\\paperw11906\\paperh16838\\margl1440\\margr1440\\margt1440\\margb1440\\widowctrl\n" % (fonts, cols, sty, "".join(lt), "".join(lo)))
    foot = "{\\footer\\pard\\plain\\qc\\fs20 \\chpgn\\par}\n" if footer else ""
    return head + foot + "\\sectd\n" + "".join(BODY) + "}\n"


def validate(rtf):
    assert rtf.count("{") - rtf.count("\\{") == rtf.count("}") - rtf.count("\\}"), "unbalanced braces"
    bad = [c for c in rtf if ord(c) > 127]
    assert not bad, "non-ascii characters: %r" % bad[:5]
    return True


def load_pages(path):
    global PAGES
    if os.path.exists(path):
        PAGES = json.load(open(path))
