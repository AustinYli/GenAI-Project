"""Alternative forms of the same missing-target retrieval query."""

from copy import deepcopy
import re

from common.bge import default_device
from common.hf import dtype_kwargs
from idea1_residual.pipeline import candidate_queries


LIST_SEPARATORS = r"(?:\s*(?:,|\band\b|\bor\b))+\s*"


def name_key(text):
    return " ".join(text.casefold().split())


def other_targets(spans, target):
    keep = name_key(spans[target].text)
    return [s for s in spans if s.kind == "target" and name_key(s.text) != keep]


def trimmed_question(question, spans, target):
    removals = []
    for span in other_targets(spans, target):
        prefix = re.search(LIST_SEPARATORS + r"$", question[:span.start], re.IGNORECASE)
        suffix = re.match(LIST_SEPARATORS, question[span.end:], re.IGNORECASE)
        start = prefix.start() if prefix else span.start
        end = span.end + suffix.end() if suffix else span.end
        removals.append((start, end))
    pieces, pos = [], 0
    for start, end in sorted(removals):
        if start > pos:
            pieces.append(question[pos:start])
        pos = max(pos, end)
    pieces.append(question[pos:])
    text = " ".join(" ".join(pieces).split())
    return re.sub(r"\s+([,.?!;:])", r"\1", text).strip(" ,")


def name_pattern(name):
    literal = r"\s+".join(re.escape(w) for w in name.split())
    return re.compile(r"(?<!\w)" + literal + r"(?!\w)", re.IGNORECASE)


def valid_rewrite(text, target, removed):
    kept = [m.span() for m in name_pattern(target).finditer(text)]
    if not kept:
        return False
    return not any(not any(lo <= m.start() and m.end() <= hi for lo, hi in kept)
                   for name in removed for m in name_pattern(name).finditer(text))


class QwenRewriter:
    def __init__(self, model_name, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        dtype = torch.bfloat16 if torch.device(self.device).type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **dtype_kwargs(dtype)).to(self.device).eval()
        self.config = deepcopy(self.model.generation_config)
        self.config.do_sample = False
        self.config.max_new_tokens = 128
        self.config.temperature = self.config.top_p = self.config.top_k = None
        self.config.pad_token_id = self.tokenizer.pad_token_id

    def rewrite(self, question, requests):
        chats = []
        for target, removed in requests:
            forbidden = ", ".join(removed) or "none"
            prompt = (f"Rewrite this question so it asks only about {target}. Keep all other details. "
                      f"Do not mention: {forbidden}. Output only the question.\n\nQuestion: {question}")
            chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                        add_generation_prompt=True)
            chats.append(chat)
        batch = self.tokenizer(chats, padding=True, add_special_tokens=False, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**batch, generation_config=self.config)
        tail = output[:, batch["input_ids"].shape[1]:].cpu()
        return [s.strip().strip('"“”').strip() for s in self.tokenizer.batch_decode(tail, skip_special_tokens=True)]


def queries_for_question(question, spans, form, keys=None, rewriter=None):
    """Return retrieval text and fallback status for each requested query key."""
    keywords = candidate_queries(spans)
    keys = list(keywords) if keys is None else keys
    result = {key: {"query": keywords[key]} for key in keys}
    targets = [(key, int(key[1:])) for key in keys if key != "kw"]
    if form == "trimmed":
        for key, i in targets:
            result[key]["query"] = trimmed_question(question, spans, i)
    elif form == "rewrite" and targets:
        requests = [(spans[i].text, list(dict.fromkeys(s.text for s in other_targets(spans, i))))
                    for _, i in targets]
        rewrites = rewriter.rewrite(question, requests)
        for (key, _), text, (target, removed) in zip(targets, rewrites, requests, strict=True):
            accepted = valid_rewrite(text, target, removed)
            result[key]["rewrite_fallback"] = not accepted
            if accepted:
                result[key]["query"] = text
    return result
