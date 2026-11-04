"""HTML emitter -> PDF via headless Chrome. Usage: python3 report/build_html.py  (reads report/pages.json if present)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content import APPENDIX_A, APPENDIX_B, B, REFS
from front import front_blocks

PAGES = json.load(open("report/pages.json")) if os.path.exists("report/pages.json") else {}


def inl(s):
    return s.replace("<hl>", '<span class="hl">').replace("</hl>", "</span>")


def toc_items():
    items = []
    for b in B:
        if b[0] == "h1":
            items.append((1, b[1]))
        elif b[0] == "h2":
            items.append((2, b[1]))
    items += [(1, "REFERENCES"), (1, "APPENDIX A: CODE ORGANISATION AND REPRODUCTION"), (1, "APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES")]
    return items


def captions():
    t, f = [], []
    for b in B:
        if b[0] == "table":
            t.append(b[1])
        if b[0] == "fig":
            f.append(b[2])
    t.append("Table B.1: Predictions written before the runs, and what happened")
    return t, f


def pg(key):
    return PAGES.get(key, "00")


def html():
    o = ['<html><head><meta charset="utf-8"><style>',
         '@page{size:A4;margin:2.54cm;@bottom-center{content:counter(page);font:10pt "Times New Roman"}}',
         'body{font:12pt/1.32 "Times New Roman",serif;color:#000}p{text-align:justify;margin:0 0 4pt}h1{font-size:14pt;margin:11pt 0 4pt;break-after:avoid}'
         'h2{font-size:12pt;margin:8pt 0 3pt;break-after:avoid}.pb{break-before:page}.c{text-align:center;margin:0}.hl{background:#ff0}'
         'table.t{border-collapse:collapse;width:100%;font-size:10pt;line-height:1.25;margin:0 0 8pt}table.t td,table.t th{border:1px solid #777;padding:2pt 5pt;vertical-align:top}'
         'table.t th{background:#D9E2F3}tr{break-inside:avoid}.tcap{text-align:center;font-weight:bold;font-size:10pt;margin:8pt 0 3pt;break-after:avoid}'
         'figure{margin:4pt 0;text-align:center;break-inside:avoid}figcaption{font-size:10pt;margin-top:2pt}ul{margin:0 0 6pt;padding-left:22pt}li{text-align:justify}'
         '.toc{display:flex;font-size:9.5pt;line-height:1.15}.toc span.d{flex:1;border-bottom:1px dotted #555;margin:0 4px 4px}.toc.l2{padding-left:16pt}.toc.l1{font-weight:bold;margin-top:2pt}'
         '.ref{padding-left:30pt;text-indent:-30pt;font-size:10.5pt;line-height:1.25;margin:0 0 3pt}.small{font-size:10pt}</style></head><body>']
    cover = True
    for b in front_blocks():
        k = b[0]
        if k == "center":
            _, t, sz, bold, ital, sa = b[:6]
            col = b[6] if len(b) > 6 and b[6] else "#000"
            sb = b[7] if len(b) > 7 else 0
            o.append('<p class="c" style="font-size:%dpt;font-weight:%s;font-style:%s;color:%s;margin:%dpt 0 %dpt">%s</p>' % (sz, "bold" if bold else "normal", "italic" if ital else "normal", col if col.startswith("#") else "#" + col, sb, sa, inl(t)))
        elif k == "cover_table":
            o.append('<table class="t" style="font-size:11pt;margin-top:6pt"><tr>' + "".join("<th>%s</th>" % h for h in b[1]) + "</tr>"
                     + "".join("<tr>" + "".join('<td style="text-align:center">%s</td>' % inl(c) for c in r) + "</tr>" for r in b[2]) + "</table>")
        elif k == "h1c":
            o.append('<h1 class="%s" style="text-align:center">%s</h1>' % ("pb" if b[2] else "", b[1]))
        elif k == "p":
            o.append("<p>%s</p>" % inl(b[1]))
        elif k == "right":
            o.append('<p style="text-align:right;margin:0 0 2pt;%s">%s</p>' % ("font-weight:bold" if b[2] else "", inl(b[1])))
        elif k == "left":
            o.append('<p style="margin:0 0 2pt">%s</p>' % inl(b[1]))
        elif k == "spacer":
            o.append('<div style="height:%dpt"></div>' % b[1])
    # contents
    o.append('<h1 class="pb" style="text-align:center">TABLE OF CONTENTS</h1>')
    for lv, t in [(1, "CERTIFICATE"), (1, "DECLARATION"), (1, "ACKNOWLEDGEMENT"), (1, "ABSTRACT")] + toc_items():
        o.append('<div class="toc l%d"><span>%s</span><span class="d"></span><span>%s</span></div>' % (lv, t, pg("H:" + t)))
    tl, fl = captions()
    o.append('<h1 style="text-align:center;margin-top:10pt">LIST OF TABLES</h1>')
    for t in tl:
        o.append('<div class="toc" style="font-size:9pt"><span>%s</span><span class="d"></span><span>%s</span></div>' % (t, pg("C:" + t.split(":")[0])))
    o.append('<h1 style="text-align:center;margin-top:10pt">LIST OF FIGURES</h1>')
    for t in fl:
        o.append('<div class="toc" style="font-size:9pt"><span>%s</span><span class="d"></span><span>%s</span></div>' % (t, pg("C:" + t.split(":")[0])))
    # body
    for b in B:
        k = b[0]
        if k == "h1":
            o.append('<h1 class="%s">%s</h1>' % ("pb" if b[2] else "", b[1]))
        elif k == "h2":
            o.append("<h2>%s</h2>" % b[1])
        elif k == "p":
            o.append("<p>%s</p>" % inl(b[1]))
        elif k == "bullets":
            o.append("<ul>" + "".join("<li>%s</li>" % inl(x) for x in b[1]) + "</ul>")
        elif k == "fig":
            o.append('<figure><img src="img/%s" style="width:%.2fin"><figcaption>%s</figcaption></figure>' % (b[1], b[3], b[2]))
        elif k == "table":
            _, cap, head, rows, widths, al = b
            o.append('<div class="tcap">%s</div><table class="t"><tr>%s</tr>%s</table>' % (
                cap, "".join('<th style="width:%.2fin">%s</th>' % (w, h) for h, w in zip(head, widths)),
                "".join("<tr>" + "".join('<td style="text-align:%s">%s</td>' % ({"l": "left", "c": "center"}[a_], inl(c)) for c, a_ in zip(r, al)) + "</tr>" for r in rows)))
    o.append('<h1>REFERENCES</h1>')
    for i, t in enumerate(REFS, 1):
        o.append('<p class="ref">[%d]&nbsp;&nbsp;%s</p>' % (i, t))
    o.append('<h1>APPENDIX A: CODE ORGANISATION AND REPRODUCTION</h1><p>%s</p>' % APPENDIX_A)
    o.append('<h1>APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES</h1><div class="tcap">Table B.1: Predictions written before the runs, and what happened</div><table class="t">'
             '<tr><th style="width:1.7in">Experiment</th><th style="width:2in">Prediction or rule</th><th>Outcome</th></tr>' + "".join("<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>" for r in APPENDIX_B) + "</table>")
    o.append("</body></html>")
    return "\n".join(o)


open("report/ORBIT_Report.html", "w").write(html())
os.system('"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=report/ORBIT_Report.pdf file://$PWD/report/ORBIT_Report.html >/dev/null 2>&1')
os.system("pdfinfo report/ORBIT_Report.pdf | grep Pages")
