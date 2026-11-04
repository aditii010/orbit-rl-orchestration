"""Find the page of every heading and caption in report/ORBIT_Report.pdf -> report/pages.json (keys H:<heading>, C:<Table x.y>/<Figure x.y>)."""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_html_lists as L

subprocess.run(["pdftotext", "-bbox-layout", "report/ORBIT_Report.pdf", "/tmp/_r.html"], check=True)
html = open("/tmp/_r.html").read()
pages = []
for pg in html.split("<page ")[1:]:
    lines = []
    for m in re.finditer(r'<line [^>]*>(.*?)</line>', pg, re.S):
        lines.append(" ".join(re.findall(r">([^<]+)</word>", m.group(1))).replace("&amp;", "&"))
    pages.append(lines)
out = {}
body_start = next(i for i, lines in enumerate(pages, 1) if lines and lines[0].strip() == "1. INTRODUCTION")   # first page of the body
front = ("CERTIFICATE", "DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT")
for key, text in L.heading_keys():
    rng = range(1, body_start) if text in front else range(body_start, len(pages) + 1)
    for i in rng:
        if any(l.strip() == text.strip() for l in pages[i - 1]):
            out[key] = str(i)
            break
for key, label in L.caption_keys():
    for i in range(body_start, len(pages) + 1):
        if any(l.startswith(label) for l in pages[i - 1]):
            out[key] = str(i)
            break
json.dump(out, open("report/pages.json", "w"), indent=0)
missing = [k for k, _ in L.heading_keys() if k not in out] + [k for k, _ in L.caption_keys() if k not in out]
print("measured", len(out), "| missing:", missing)
