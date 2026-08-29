#!/usr/bin/python3

################################################################################
# Some INSPIRE abstracts carry raw, unstripped HTML/MathML markup (e.g.
# Elsevier-sourced records via SCOAP3: "<math><mrow><mi>p</mi><mo>=</mo>...")
# -- about 5% of the corpus. Left in, tag names like "mi"/"mo"/"mrow" get fed
# straight into SPECTER as if they were real words, and show up as bogus
# top terms in any downstream TF-IDF. Strip tags, keep the text content.
################################################################################

import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_markup(text):
    if not text:
        return text
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()
