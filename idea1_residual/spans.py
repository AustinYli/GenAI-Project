"""Split a question into spans: the named things it asks about (targets) and the details (descriptive).

  target       companies, people, products, places, events, news outlets ...
               -> may be removed from the second search when covered
  descriptive  noun phrases, dates, amounts ("higher revenue", "the year 2023")
               -> always kept in second-search queries

News outlet names are matched from a list taken from the corpus (spaCy often
mislabels names like "The Verge"), then spaCy entities, then noun chunks.
"""
import re
from dataclasses import asdict, dataclass
from typing import List

TARGET_LABELS = {"ORG", "PERSON", "PRODUCT", "GPE", "EVENT", "NORP", "WORK_OF_ART", "FAC", "LOC", "LAW", "OUTLET"}
SKIP_POS = {"DET", "PRON", "PUNCT", "CCONJ", "ADP", "PART", "AUX", "SCONJ", "SPACE"}
WH = {"which", "what", "who", "whom", "whose", "when", "where", "why", "how"}
CONTENT_POS = {"NOUN", "PROPN", "ADJ", "NUM", "X"}


@dataclass
class Span:
    text: str
    start: int
    end: int
    label: str
    kind: str  # "target" | "descriptive"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Span(**d)


class SpanExtractor:
    def __init__(self, outlets: List[str], spacy_model: str = "en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(spacy_model)
        self.outlet_res = [
            re.compile(r"(?<![\w])" + re.escape(o) + r"(?![\w])") for o in sorted(set(outlets), key=len, reverse=True)
        ]

    def __call__(self, question: str) -> List[Span]:
        doc = self.nlp(question)
        taken = [False] * (len(question) + 1)
        spans: List[Span] = []

        def free(s, e):
            return not any(taken[s:e])

        def add(s, e, label, kind):
            # trim whitespace / quotes / trailing punctuation
            while s < e and question[s] in " \t\n'\"“”‘’(":
                s += 1
            while e > s and question[e - 1] in " \t\n'\"“”‘’),.?!:;":
                e -= 1
            if e - s < 2 or not free(s, e):
                return
            for i in range(s, e):
                taken[i] = True
            spans.append(Span(question[s:e], s, e, label, kind))

        for rx in self.outlet_res:
            for m in rx.finditer(question):
                add(m.start(), m.end(), "OUTLET", "target")

        for ent in doc.ents:
            s, e = ent.start_char, ent.end_char
            if not free(s, e):
                continue
            kind = "target" if ent.label_ in TARGET_LABELS else "descriptive"
            # drop a leading "the"
            if ent[0].lower_ == "the" and len(ent) > 1:
                s = ent[1].idx
            add(s, e, ent.label_, kind)

        for nc in doc.noun_chunks:
            group = []
            for tok in list(nc) + [None]:
                ok = (tok is not None and tok.pos_ not in SKIP_POS and tok.lower_ not in WH
                      and free(tok.idx, tok.idx + len(tok.text)))
                if ok:
                    group.append(tok)
                    continue
                if group and any(t.pos_ in CONTENT_POS for t in group):
                    add(group[0].idx, group[-1].idx + len(group[-1].text), "NP", "descriptive")
                group = []

        spans.sort(key=lambda sp: sp.start)
        return spans


def keyword_query(spans: List[Span], drop: List[Span] = ()) -> str:
    """Spans in question order, minus the ones in `drop`."""
    dropped = {(d.start, d.end) for d in drop}
    return " ".join(s.text for s in spans if (s.start, s.end) not in dropped)
