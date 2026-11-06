from content import ABSTRACT, KEYWORDS, SUBTITLE, TITLE

PH = lambda s: "<hl>[%s]</hl>" % s


def front_blocks():
    f = []
    a = f.append
    a(("center", "THAPAR INSTITUTE OF ENGINEERING AND TECHNOLOGY", 16, True, False, 4))
    a(("center", "(Deemed to be University), Patiala, Punjab 147004", 11, False, False, 4))
    a(("center", "Department of Electronics and Communication Engineering", 13, True, False, 36))
    a(("center", "A PROJECT REPORT ON", 12, False, False, 18))
    a(("center", "ORBIT", 40, True, False, 4, "1F3864"))
    a(("center", "Reinforcement Learning for Cost-Aware Orchestration of Multi-Agent LLM Systems", 18, True, False, 8))
    a(("center", SUBTITLE, 13, False, True, 34))
    a(("center", "Submitted in partial fulfilment of the requirements of the laboratory evaluation for", 11, False, False, 2))
    a(("center", "Course UCS760 (Reinforcement Learning and Conversational AI)", 12, True, False, 30))
    a(("cover_table",
       ["Submitted to", "Submitted by"],
       [[PH("Name of faculty member"), "<b>Aditi Sikarwar</b>"], [PH("Designation"), "Roll No.: " + PH("Roll number")],
        ["Department of ECE, TIET Patiala", "Electronics and Communication Engineering, Class of 2027"]]))
    a(("center", "November 2026", 12, True, False, 2, None, 30))
    a(("center", "Patiala, Punjab", 11, False, False, 0))
    a(("h1c", "CERTIFICATE", True))
    a(("p", "This is to certify that the project report entitled <b>\"" + TITLE + "\"</b>, submitted by <b>Aditi Sikarwar</b> (Roll No. " + PH("Roll number") + "), a student of the "
           "Department of Electronics and Communication Engineering, Thapar Institute of Engineering and Technology, Patiala, in partial fulfilment of the requirements of the laboratory "
           "evaluation for course UCS760, is a bonafide record of the work carried out by the student under my guidance and supervision during October and November 2026. The work has "
           "not been submitted, in part or in full, for the award of any other degree, diploma or certificate."))
    a(("spacer", 28))
    a(("right", PH("Name of supervisor"), True))
    a(("right", PH("Designation") + ", Department of Electronics and Communication Engineering", False))
    a(("right", "Thapar Institute of Engineering and Technology, Patiala", False))
    a(("spacer", 10))
    a(("left", "Date: " + PH("Date") + "          Place: Patiala"))
    a(("h1c", "DECLARATION", False))
    a(("p", "I hereby declare that the project report entitled <b>\"" + TITLE + "\"</b> is a record of my own project work, carried out for the laboratory evaluation of course UCS760 under the "
           "guidance of " + PH("Name of supervisor") + ". All sources of information, published work and software libraries that I used are acknowledged in the text and in the references."))
    a(("p", "<b>Disclosure of AI assistance.</b> The software, experiments and drafts of this report were developed with the help of an AI coding assistant (Claude Code, Anthropic), which wrote "
           "and ran code and drafted text under my direction. I defined the project, chose the research questions and evaluation plan, reviewed the results and the claims made from them, and "
           "take responsibility for the contents. Every experiment is recorded in the project decision log, including the mistakes that were found and corrected."))
    a(("right", "<b>Aditi Sikarwar</b>, Roll No. " + PH("Roll number") + "          Date: " + PH("Date"), False))
    a(("h1c", "ACKNOWLEDGEMENT", True))
    a(("p", "I thank my supervisor, " + PH("Name of supervisor") + ", for the guidance and feedback that shaped this project, and for the freedom to treat a negative result as a result. I am grateful to the "
           "Department of Electronics and Communication Engineering and to Thapar Institute of Engineering and Technology for the laboratory facilities. I acknowledge the open-source tools on "
           "which this work stands (PyTorch, Ollama and the Groq inference service) and the authors of the papers in the references. Finally, I thank my family and friends for their support."))
    a(("h1c", "ABSTRACT", False))
    for t in ABSTRACT:
        a(("p", t))
    a(("p", "<b>Keywords:</b> " + KEYWORDS + "."))
    return f
