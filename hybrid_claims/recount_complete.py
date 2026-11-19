"""Recount complete coverage from serialized selections and raw benchmark files."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


REMOVE = str.maketrans('', '', ' \n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def matching_ranks(fact, rendered_chunks):
    needle = fact.translate(REMOVE)
    if not needle:
        raise ValueError('Empty evidence fact')
    return [rank for rank, body in enumerate(rendered_chunks, 1) if needle in body]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selections', type=Path)
    parser.add_argument('chunks', type=Path)
    parser.add_argument('queries', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with gzip.open(args.selections, 'rt', encoding='utf-8') as f:
        selections = json.load(f)
    with args.chunks.open(encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    article_ids = [chunk['article_id'] for chunk in chunks]
    bodies = [chunk['text'].translate(REMOVE) for chunk in chunks]
    queries = read_json(args.queries)
    facts = {str(q): [e['fact'] for e in row['evidence_list']] for q, row in enumerate(queries)
             if row['question_type'] != 'null_query'}

    audit = {}
    for system, cuts in selections.items():
        audit[system] = {}
        for k, by_qid in cuts.items():
            if by_qid.keys() != facts.keys():
                raise ValueError(f'Question IDs differ for {system} at K={k}')
            coverage, issues = {}, Counter()
            examples = {}
            for qid, ids in by_qid.items():
                if any(type(cid) is not int or not 0 <= cid < len(chunks) for cid in ids):
                    raise ValueError(f'Invalid chunk ID for {system} at K={k}, qid {qid}')
                per_article = Counter(article_ids[cid] for cid in ids)
                violations = {
                    'short': len(ids) < int(k),
                    'long': len(ids) > int(k),
                    'repeat': len(ids) != len(set(ids)),
                    **{f'over_cap{cap}': max(per_article.values(), default=0) > cap for cap in (1, 2, 3)},
                }
                for issue, present in violations.items():
                    if present:
                        issues[issue] += 1
                        examples.setdefault(issue, qid)
                top = [bodies[cid] for cid in ids[:int(k)]]
                coverage[qid] = [matching_ranks(fact, top) for fact in facts[qid]]
            audit[system][k] = {'fact_ranks': coverage, 'issues': issues, 'example_qids': examples}
    args.output.write_text(json.dumps(audit, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    main()
