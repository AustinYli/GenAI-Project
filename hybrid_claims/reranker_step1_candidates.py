"""Build the fixed lexical-claim and whole-question reranker candidate lists."""

import argparse
import gzip
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

from common import paths
from common.data import gold_facts, is_answerable, load_chunks, load_queries, outlet_names, split_queries
from common.metrics import calculate_metrics, facts_found
from hybrid_claims import run
from hybrid_claims.split_training_generation import read_json, write_json


OUT = Path(paths.OUT) / 'reranker'
DENSE_CACHE = Path(paths.CACHE) / f'dense_ranking_{run.MODEL}_fp32.json'
FOLDS = Path(paths.OUT) / 'split_training' / 'cv' / 'folds.json'
CANDIDATES = OUT / 'step1_candidates.json'
PREVIOUS_AUDIT = OUT / 'previous_ablation_audit.json'
DEPTH = 60
WHOLE_DEPTH = 100
RRF_K = 60
K1 = 0.9
B = 0.4
TOP_K = 10
ARTICLE_CAP = 2
DEMO_IDS = {164, 654, 1006, 1966}


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def eligible_questions(queries, folds):
    qids = [q['qid'] for q in queries if is_answerable(q) and q['qid'] not in DEMO_IDS]
    if (len(qids) != 2251 or folds['eligible_qids'] != qids
            or set(folds['excluded_demo_qids']) != DEMO_IDS):
        raise ValueError('CV eligibility differs from the fixed 2,251 non-demonstration answerable questions')
    return qids


def candidate_design():
    return {
        'schema': 1, 'question_count': 2251, 'per_list_depth': DEPTH, 'whole_retriever_depth': WHOLE_DEPTH,
        'rrf_k': RRF_K, 'bm25_k1': K1, 'bm25_b': B, 'article_cap': ARTICLE_CAP, 'final_top_k': TOP_K,
        'backfill': False, 'dense_model': 'BAAI/bge-large-en-v1.5', 'dense_dtype': 'fp32',
        'whole_query': 'BGE and BM25, fused before truncation', 'claim_query': 'one BM25 list per rule claim',
        'source_sha256': {key: digest(path) for key, path in (
            ('queries', paths.QUERIES_JSON), ('chunks', paths.CHUNKS_JSONL), ('dense', DENSE_CACHE),
            ('folds', FOLDS), ('retrieval_rules', run.__file__),
        )},
    }


def validate_candidates(payload, qids, queries):
    rows = payload['rows']
    if [row['qid'] for row in rows] != qids or payload['bm25_index_seconds'] < 0:
        raise ValueError('Stored candidates do not match the eligible question IDs or index timing')
    for row in rows:
        question = queries[row['qid']]
        if (not row['texts'] or row['texts'][0] != question['query'] or row['question_type'] != question['question_type']
                or row['candidate_seconds'] < 0 or len(row['texts']) != len(row['lists'])
                or any(len(ids) != DEPTH or len(set(ids)) != DEPTH for ids in row['lists'])):
            raise ValueError(f"Invalid candidate lists for question {row['qid']}")


def prepare_candidates():
    queries = load_queries()
    qids = eligible_questions(queries, read_json(FOLDS))
    design = candidate_design()
    if CANDIDATES.is_file():
        payload = read_json(CANDIDATES)
        if payload['design'] != design:
            raise ValueError('Stored Step 1 candidates use a different retrieval design or source revision')
        validate_candidates(payload, qids, queries)
        return payload

    chunks = load_chunks()
    outlets = sorted(outlet_names(chunks), key=len, reverse=True)
    dense = read_json(DENSE_CACHE)['ids']
    started = perf_counter()
    bm25 = run.BM25([run.chunk_doc(chunk) for chunk in chunks], K1, B)
    index_seconds = perf_counter() - started
    rows = []
    for qid in qids:
        question = queries[qid]
        query = question['query']
        started = perf_counter()
        claims = run.claims(query, outlets)
        whole = run.rrf([dense[qid][:WHOLE_DEPTH], bm25.search(query, WHOLE_DEPTH)], k=RRF_K)
        lists = [whole[:DEPTH], *(bm25.search(claim, DEPTH) for claim in claims)]
        rows.append({'qid': qid, 'question_type': question['question_type'], 'texts': [query, *claims],
                     'lists': lists, 'candidate_seconds': perf_counter() - started})
    payload = {'design': design, 'rows': rows, 'bm25_index_seconds': index_seconds}
    validate_candidates(payload, qids, queries)
    write_json(CANDIDATES, payload)
    return payload


def summarize(ranks, gold):
    found = [facts_found(texts, facts, TOP_K) for texts, facts in zip(ranks, gold, strict=True)]
    complete = sum(all(flags) for flags in found)
    return {
        'n': len(found), 'complete_count': complete, 'complete@10': complete / len(found),
        'recall@10': sum(sum(flags) / len(flags) for flags in found) / len(found),
        'short_top10': sum(len(texts) < TOP_K for texts in ranks),
        **calculate_metrics(ranks, gold),
    }


def compare(reference, comparison):
    return {
        'rankings_changed': sum(a != b for a, b in zip(reference, comparison, strict=True)),
        'original_prefix_preserved': sum(b[:len(a)] == a for a, b in zip(reference, comparison, strict=True)),
    }


def write_previous_ablation_audit(payload):
    """Compare current A60 and untruncated selection against the archived rule baseline."""
    queries, chunks = load_queries(), load_chunks()
    step1_rows = {row['qid']: row for row in payload['rows']}
    eligible = list(step1_rows)
    dev, test = split_queries(queries, seed=13)
    dev, test = ([q for q in split if is_answerable(queries[q])] for split in (dev, test))
    if (len(dev), len(test)) != (681, 1574) or not set(test).issubset(step1_rows):
        raise ValueError('The original answerable test cohort does not match Step 1 eligibility')
    old_exact_dir = Path(paths.OUT) / 'llm_decomp_vs_hyde'
    old_exact = {row['qid']: row['system']['B']['ids@10'] for split in ('dev', 'test')
                 for row in read_json(old_exact_dir / f'{split}_per_question.json')}
    if set(old_exact) != set(dev) | set(test):
        raise ValueError('Archived exact-K baseline does not cover the original answerable cohorts')
    old_short_file = Path(paths.OUT) / 'hybrid_claims' / 'verification' / 'chunk_ids.json.gz'
    with gzip.open(old_short_file, 'rt', encoding='utf-8') as stream:
        old_short = {int(q): ids for q, ids in json.load(stream)['original:claims']['10'].items()}
    if set(old_short) != set(old_exact):
        raise ValueError('Archived finite candidate baseline does not cover the same cohorts')

    dense = read_json(DENSE_CACHE)['ids']
    bm25 = run.BM25([run.chunk_doc(chunk) for chunk in chunks], K1, B)
    ranks = {name: {} for name in ('A60_no_backfill', 'uncropped_no_backfill', 'archived_no_backfill',
                                   'archived_exact_k')}
    for qid, row in step1_rows.items():
        lists = row['lists']
        whole = run.rrf([dense[qid][:WHOLE_DEPTH], bm25.search(row['texts'][0], WHOLE_DEPTH)], k=RRF_K)
        if lists[0] != whole[:DEPTH]:
            raise ValueError(f'Step 1 whole-question head differs from rebuilt original for question {qid}')
        ranks['A60_no_backfill'][qid] = run.pick(lists, TOP_K, chunks, ARTICLE_CAP)
        ranks['uncropped_no_backfill'][qid] = run.pick([whole, *lists[1:]], TOP_K, chunks, ARTICLE_CAP)
        ranks['archived_no_backfill'][qid] = old_short[qid]
        ranks['archived_exact_k'][qid] = old_exact[qid]

    def audit_cohort(qids):
        gold = [gold_facts(queries[qid]) for qid in qids]
        ordered = {name: [values[qid] for qid in qids] for name, values in ranks.items()}
        stats = {name: summarize([[chunks[c]['text'] for c in ids] for ids in lists], gold)
                 for name, lists in ordered.items()}
        counts = {
            'A60_vs_uncropped': compare(ordered['uncropped_no_backfill'], ordered['A60_no_backfill']),
            'reconstructed_uncropped_vs_archive': compare(ordered['archived_no_backfill'],
                                                            ordered['uncropped_no_backfill']),
            'archived_exact_k_vs_no_backfill': compare(ordered['archived_no_backfill'], ordered['archived_exact_k']),
            'A60_vs_archived_exact_k': compare(ordered['archived_exact_k'], ordered['A60_no_backfill']),
        }
        return {'systems': stats, 'rank_comparisons': counts}

    test_audit = audit_cohort(test)
    cross_file = Path(paths.OUT) / 'replicate' / 'bge-large-en-v1.5_rerank_retrieval_test.json'
    cross = read_json(cross_file)
    if len(cross) != len(queries) or any(cross[q]['query'] != queries[q]['query'] for q in test):
        raise ValueError('Dense-only cross-encoder archive does not match the fixed test questions')
    reranked = summarize([[candidate['text'] for candidate in cross[q]['retrieval_list']] for q in test],
                         [gold_facts(queries[q]) for q in test])
    exact = test_audit['systems']['archived_exact_k']
    result = {
        'finding': f"The archived 31.68% is MAP@10 for the rule-split baseline, which had "
                   f"{exact['complete@10']:.2%} complete@10 ({exact['complete_count']}/{len(test)}) and no cross-encoder. "
                   f"The archived cross-encoder instead reranked dense BGE's top 20 and had "
                   f"{reranked['complete@10']:.2%} complete@10 on the same test questions.",
        'legacy_fixed_test': {'n': len(test), 'seed': 13, 'qids': test, 'rule_exact_k': exact,
                              'rule_no_backfill': test_audit['systems']['archived_no_backfill'],
                              'dense_cross_encoder': reranked,
                              'dense_cross_encoder_config': 'BGE top 20, bge-reranker-large, no cap, keep 10',
                              'rule_archive_sources': [str(old_exact_dir / 'test_per_question.json'), str(old_short_file)],
                              'cross_encoder_archive_source': str(cross_file)},
        'step1_same_test': test_audit,
        'step1_all_eligible': audit_cohort(eligible),
        'design': payload['design'],
    }
    write_json(PREVIOUS_AUDIT, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true', help='Also compare the previous baseline archives')
    args = parser.parse_args()
    payload = prepare_candidates()
    print(f"Saved candidates for {len(payload['rows']):,} questions to {CANDIDATES}", flush=True)
    if args.audit:
        audit = write_previous_ablation_audit(payload)
        print(audit['finding'], flush=True)


if __name__ == '__main__':
    main()
