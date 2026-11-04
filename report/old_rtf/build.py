"""Assemble the report: python3 report/build.py  ->  report/ORBIT_Report.rtf (+ lists for the contents page)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtf as R
import part1, part2, part3, part4

R.load_pages("report/toc_pages.json")
part1.build()
n_front = len(R.BODY)
part2.build()
part3.build()
part4.build()

heads = [(lv, t) for lv, t in R.HEAD]
caps = list(R.CAPS)
# contents pages are built into a separate list and spliced where %%CONTENTS%% sits
save = R.BODY[:]
R.BODY.clear()
R.add("\\pard\\pagebb\\s1\\outlinelevel0\\keepn\\sb240\\sa200\\qc\\plain\\b\\fs32 TABLE OF CONTENTS\\par\n")
for lv, t in heads:
    if lv == 1:
        R.toc_line(t, "H:" + t, bold=True)
    elif lv == 2:
        R.toc_line(t, "H:" + t, indent=360)
R.add("\\pard\\pagebb\\s1\\outlinelevel0\\keepn\\sb240\\sa200\\qc\\plain\\b\\fs32 LIST OF TABLES\\par\n")
for kind, label, text in caps:
    if kind == "table":
        R.toc_line("%s: %s" % (label, text), "C:" + label, size=21)
R.add("\\pard\\pagebb\\s1\\outlinelevel0\\keepn\\sb240\\sa200\\qc\\plain\\b\\fs32 LIST OF FIGURES\\par\n")
for kind, label, text in caps:
    if kind == "figure":
        R.toc_line("%s: %s" % (label, text), "C:" + label, size=21)
contents = "".join(R.BODY)
R.BODY[:] = [contents if x == "%%CONTENTS%%" else x for x in save]
rtf = R.document()
R.validate(rtf)
open("report/ORBIT_Report.rtf", "w").write(rtf)
json.dump({"heads": heads, "caps": [(k, l, t) for k, l, t in caps]}, open("report/structure.json", "w"))
words = sum(len(x.split()) for x in R.BODY)
print("RTF bytes:", len(rtf), "| headings:", len(heads), "| tables:", sum(1 for c in caps if c[0] == "table"), "| figures:", sum(1 for c in caps if c[0] == "figure"))
