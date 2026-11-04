"""Keys used by the contents pages (shared by measure.py and the builders)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content import B


def heading_keys():
    keys = [("H:" + t, t) for t in ("CERTIFICATE", "DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT")]
    for b in B:
        if b[0] in ("h1", "h2"):
            keys.append(("H:" + b[1], b[1]))
    keys += [("H:REFERENCES", "REFERENCES"), ("H:APPENDIX A: CODE ORGANISATION AND REPRODUCTION", "APPENDIX A: CODE ORGANISATION AND REPRODUCTION"),
             ("H:APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES", "APPENDIX B: PRE-REGISTERED PREDICTIONS AND OUTCOMES")]
    return keys


def caption_keys():
    ks = []
    for b in B:
        if b[0] == "table":
            ks.append(("C:" + b[1].split(":")[0], b[1].split(":")[0] + ":"))
        if b[0] == "fig":
            ks.append(("C:" + b[2].split(":")[0], b[2].split(":")[0] + ":"))
    ks.append(("C:Table B.1", "Table B.1:"))
    return ks
