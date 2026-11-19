"""Coverage scorers: how much do the first-pass chunks D1 "explain" each span of the question?

Every scorer returns one gain per span. High gain = covered, low gain = missing.

  word         fraction of the span's content words that appear in D1 (the word-absence baseline)
  llm_inplace  sum over the span's tokens of  log p(t | instr, D1, question prefix) - log p(t | instr, question prefix)
               (read in place while the answering LLM reads the question; one forward pass each)
  llm_inplace_mean  same, averaged over the span's tokens (the first formula from the discussion)
  llm_probe    mean over probe sentences of  log p(span | D1, probe) - log p(span | probe)
               (each span scored after the same neutral sentence, so word order doesn't matter)
  mlm          sum over the span's tokens of  log p(t | D1, question with span masked) - log p(t | question with span masked)
               (masked encoder such as ModernBERT: sees both sides of the blank)
"""
from common.hf import dtype_kwargs
import copy
import re
from typing import List, Optional, Sequence

from .spans import Span

INSTRUCTION = ("Answer the question using only the passages below. "
               "If they don't contain the answer, say \"Insufficient information\".")
PROBES = ["The passages above discuss:", "The passages above contain information about:"]

_STOP = set("""a an the of and or in on at to for from by with about as is are was were be been being do does did
has have had that this these those it its their his her which what who whom whose when where why how both
according reported report article articles news""".split())


def _words(s: str):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def format_passages(passages: Sequence[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(passages))


class WordScorer:
    name = "word"

    def score(self, question: str, spans: List[Span], passages: Sequence[str]) -> List[float]:
        vocab = set(_words(" ".join(passages)))
        out = []
        for s in spans:
            ws = _words(s.text)
            out.append(sum(w in vocab for w in ws) / len(ws) if ws else 1.0)
        return out


class _CausalLM:
    def __init__(self, model_name: str, device: Optional[str] = None, dtype: str = "bf16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **dtype_kwargs(td)).to(self.device).eval()
        self.base = getattr(self.model, "model", None)
        self.head = self.model.get_output_embeddings()

    def hidden(self, ids, past=None):
        """Last hidden states (and cache) without computing logits for every position."""
        if self.base is not None:
            out = self.base(input_ids=ids, past_key_values=past, use_cache=True)
            return out.last_hidden_state, out.past_key_values
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True, output_hidden_states=True)
        return out.hidden_states[-1], out.past_key_values

    def logprobs_at(self, h):
        return self.head(h).float().log_softmax(-1)

    def chat(self, user: str) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                add_generation_prompt=True)
        return user


class LLMInPlaceScorer(_CausalLM):
    def __init__(self, model_name: str, reduce: str = "sum", **kw):
        super().__init__(model_name, **kw)
        self.reduce = reduce
        self.name = "llm_inplace" if reduce == "sum" else "llm_inplace_mean"

    def prompt(self, question: str, passages: Optional[Sequence[str]]):
        user = INSTRUCTION
        if passages:
            user += "\n\nPassages:\n" + format_passages(passages)
        user += "\n\nQuestion: " + question
        text = self.chat(user)
        q_start = text.rindex("Question: " + question) + len("Question: ")
        return text, q_start

    def question_token_logprobs(self, text: str, q_start: int, q_end: int):
        """[(char_start, char_end, logprob)] for the tokens of the question (positions relative to the question)."""
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        first = next(i for i, (s, e) in enumerate(offs) if e > q_start)
        last = max(i for i, (s, e) in enumerate(offs) if s < q_end)
        t = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            h, _ = self.hidden(t)
            lp = self.logprobs_at(h[0, first - 1:last])  # predicts tokens first..last
        res = []
        for j, i in enumerate(range(first, last + 1)):
            s, e = offs[i]
            res.append((s - q_start, e - q_start, lp[j, ids[i]].item()))
        return res

    def score(self, question: str, spans: List[Span], passages: Sequence[str]) -> List[float]:
        sums = []
        for ctx in (passages, None):
            text, q0 = self.prompt(question, ctx)
            toks = self.question_token_logprobs(text, q0, q0 + len(question))
            per = []
            for sp in spans:
                vals = [lp for s, e, lp in toks if s < sp.end and e > sp.start]
                per.append((sum(vals), max(1, len(vals))))
            sums.append(per)
        out = []
        for (w, n_w), (wo, n_wo) in zip(*sums):
            out.append(w - wo if self.reduce == "sum" else w / n_w - wo / n_wo)
        return out


class LLMProbeScorer(_CausalLM):
    name = "llm_probe"

    def __init__(self, model_name: str, probes: Sequence[str] = PROBES, **kw):
        super().__init__(model_name, **kw)
        self.probes = list(probes)

    def _prefix(self, text: str):
        torch = self.torch
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        with torch.inference_mode():
            h, past = self.hidden(ids)
            last = self.logprobs_at(h[0, -1:])[0]
        return past, last

    def _cont(self, past, last_lp, cont_ids: List[int]) -> float:
        torch = self.torch
        total = last_lp[cont_ids[0]].item()
        if len(cont_ids) > 1:
            pk = copy.deepcopy(past)
            with torch.inference_mode():
                h, _ = self.hidden(torch.tensor([cont_ids[:-1]], device=self.device), past=pk)
                lp = self.logprobs_at(h[0])
            total += sum(lp[i, cont_ids[i + 1]].item() for i in range(len(cont_ids) - 1))
        return total

    def score(self, question: str, spans: List[Span], passages: Sequence[str]) -> List[float]:
        ctx = "Passages:\n" + format_passages(passages) + "\n\n"
        gains = [0.0] * len(spans)
        conts = [self.tok(" " + sp.text, add_special_tokens=False).input_ids for sp in spans]
        for probe in self.probes:
            pw, lw = self._prefix(ctx + probe)
            po, lo = self._prefix(probe)
            for i, c in enumerate(conts):
                if c:
                    gains[i] += (self._cont(pw, lw, c) - self._cont(po, lo, c)) / len(self.probes)
            del pw, po
        return gains


class MaskedLMScorer:
    """h(s) with a masked encoder (default ModernBERT-large, 8,192-token context)."""

    name = "mlm"

    def __init__(self, model_name: str = "answerdotai/ModernBERT-large", device: Optional[str] = None,
                 dtype: str = "bf16", max_length: Optional[int] = None):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, **dtype_kwargs(td)).to(self.device).eval()
        self.max_length = max_length or min(8192, int(getattr(self.model.config, "max_position_embeddings", 512)))

    def _span_logprob(self, prefix_ids: List[int], q_ids: List[int], pos: List[int]) -> float:
        torch = self.torch
        masked = list(q_ids)
        for p in pos:
            masked[p] = self.tok.mask_token_id
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        ids = [cls] + (prefix_ids + [sep] if prefix_ids else []) + masked + [sep]
        offset = len(ids) - len(masked) - 1
        with torch.inference_mode():
            logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0]
            lp = logits[[offset + p for p in pos]].float().log_softmax(-1)
        return sum(lp[j, q_ids[p]].item() for j, p in enumerate(pos))

    def score(self, question: str, spans: List[Span], passages: Sequence[str]) -> List[float]:
        enc = self.tok(question, add_special_tokens=False, return_offsets_mapping=True)
        q_ids, offs = enc["input_ids"], enc["offset_mapping"]
        budget = self.max_length - len(q_ids) - 3
        ctx_ids = self.tok("\n\n".join(p.strip() for p in passages), add_special_tokens=False)["input_ids"][:budget]
        out = []
        for sp in spans:
            pos = [i for i, (s, e) in enumerate(offs) if s < sp.end and e > sp.start]
            if not pos:
                out.append(0.0)
                continue
            out.append(self._span_logprob(ctx_ids, q_ids, pos) - self._span_logprob([], q_ids, pos))
        return out


def make_scorer(name: str, llm: str, mlm: str, device=None):
    if name == "word":
        return WordScorer()
    if name == "llm_inplace":
        return LLMInPlaceScorer(llm, reduce="sum", device=device)
    if name == "llm_inplace_mean":
        return LLMInPlaceScorer(llm, reduce="mean", device=device)
    if name == "llm_probe":
        return LLMProbeScorer(llm, device=device)
    if name == "mlm":
        return MaskedLMScorer(mlm, device=device)
    raise ValueError(name)
