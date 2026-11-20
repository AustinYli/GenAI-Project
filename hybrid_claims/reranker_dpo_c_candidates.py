"""Replace the Step 1 claim lists with lexical lists for frozen out-of-fold splits."""

from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter

from common.data import load_chunks, load_queries
from hybrid_claims import llm_compare, run
from hybrid_claims.reranker_step1_candidates import (
    B, CANDIDATES as STEP1_CANDIDATES, DEPTH, FOLDS, K1, OUT, digest, eligible_questions,
    validate_candidates,
)
from hybrid_claims.split_training_cv_metrics import gold_articles, raw_greedy, validate_folds
from hybrid_claims.split_training_dpo import CV_PRECISIONS
from hybrid_claims.split_training_generation import read_json, write_json


CANDIDATES = OUT / 'dpo_c_candidates.json'
OOF_FOLDS = ('fold_1', 'fold_2')
ARMS = ('zero_base', 'zero_dpo', 'few_base', 'few_dpo')


def cv_sources():
    paths = {}
    for fold in OOF_FOLDS:
        for prompt in CV_PRECISIONS:
            arm = FOLDS.parent / fold / f'3b_{prompt}'
            for role, filename in (('base', 'test_base.json'), ('dpo', 'test_trained.json'),
                                   ('selection', 'selection.json')):
                paths[fold, prompt, role] = arm / filename
    return paths


def cached_lexical_ranks(database):
    if database is None:
        return {}
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        return {text: json.loads(ranking) for text, ranking in
                conn.execute('SELECT text, ranking FROM rankings WHERE scope = ?', ('claim_lexical',))}


def raw_decompositions(folds, sources):
    arms = {arm: {} for arm in ARMS}
    for fold in OOF_FOLDS:
        test = folds[fold]['test']
        for prompt, precision in CV_PRECISIONS.items():
            source = sources[fold, prompt, 'selection']
            decision = read_json(source)
            candidates = [point for trial in decision['trials'] for point in trial['validation']]
            selected = decision['selected']
            if (not candidates or selected != max(candidates, key=lambda p: (p['complete@10'], -p['epoch'], -p['lr']))
                    or selected['precision'] != precision or not Path(selected['path']).resolve().is_relative_to(source.parent)):
                raise ValueError(f'Archived {fold}/{prompt} selection does not match its fold or precision')
            for role, checkpoint in (('base', None), ('dpo', selected['path'])):
                target = arms[f'{prompt}_{role}']
                if target.keys() & set(test):
                    raise ValueError('An evaluation question appears in multiple outer folds')
                path = sources[fold, prompt, role]
                raw = raw_greedy(path, test, '3b_' + prompt, precision, checkpoint)
                timings = read_json(path)['rows']
                for qid, text in raw.items():
                    seconds = timings[str(qid)]['seconds']
                    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
                        raise ValueError(f'Invalid archived split time for {fold}/{prompt}/{role}, qid {qid}')
                    target[qid] = text, seconds
    return arms


def valid_ids(ids, n_chunks):
    return (len(ids) == DEPTH and len(set(ids)) == DEPTH
            and all(type(cid) is int and 0 <= cid < n_chunks for cid in ids))


def validate(payload, step1, n_chunks, decompositions):
    if len(payload['rows']) != len(step1):
        raise ValueError('Candidate archive must cover all eligible questions')
    for row, frozen in zip(payload['rows'], step1, strict=True):
        qid = frozen['qid']
        if row['qid'] != qid or row['question_type'] != frozen['question_type'] or set(row['systems']) != set(ARMS):
            raise ValueError(f'Out-of-fold archive does not match the Step 1 question or expected arms for {qid}')
        for arm, item in row['systems'].items():
            raw, split_seconds = decompositions[arm][qid]
            splits = llm_compare.valid_lines(raw)
            expected_texts = [frozen['texts'][0], *splits] if splits else frozen['texts']
            seconds = item['candidate_seconds']
            if (item['texts'] != expected_texts or item['fallback'] != (not splits)
                    or not math.isfinite(seconds) or seconds < 0
                    or item['split_seconds'] != split_seconds
                    or len(item['lists']) != len(expected_texts) or item['lists'][0] != frozen['lists'][0]
                    or (not splits and item['lists'] != frozen['lists'])
                    or not all(valid_ids(ids, n_chunks) for ids in item['lists'])):
                raise ValueError(f'Invalid {arm} candidate lists for qid {qid}')


def prepare_candidates():
    queries, chunks = load_queries(), load_chunks()
    manifest = read_json(FOLDS)
    qids = eligible_questions(queries, manifest)
    folds = validate_folds(manifest, gold_articles(queries))
    step1 = read_json(STEP1_CANDIDATES)
    validate_candidates(step1, qids, queries)
    sources = cv_sources()
    archives = sorted((FOLDS.parent / 'retrieval').glob('rankings_*.sqlite3'))
    if len(archives) > 1:
        raise ValueError('Multiple old lexical ranking archives found')
    database = archives[0] if archives else None
    design = {
        **step1['design'],
        'claim_query': 'one lexical BM25 list per surviving out-of-fold sub-question',
        'fallback': 'the exact Step 1 rule claims and lexical ranks when no sub-question survives',
        'candidate_timing': 'Takes the archived whole-query head, filters cached greedy lines, and includes '
                            'actual BM25 search for new text and in-memory lookup for cache hits. Excludes LLM '
                            'generation, original whole-query retrieval, and dataset and index setup.',
        'outer_folds': list(OOF_FOLDS), 'arms': list(ARMS), 'base_precision': CV_PRECISIONS,
        'frozen_archive_sha256': {'step1_candidates': digest(STEP1_CANDIDATES), 'folds': digest(FOLDS),
                                  'valid_lines': digest(llm_compare.__file__),
                                  'archived_lexical_cache': digest(database) if database is not None else None,
                                  **{'/'.join(key): digest(path) for key, path in sources.items()}},
    }
    decompositions = raw_decompositions(folds, sources)
    if any(set(raw) != set(qids) for raw in decompositions.values()):
        raise ValueError('All four greedy decomposition arms must cover the frozen cohort once')
    if CANDIDATES.is_file():
        result = read_json(CANDIDATES)
        if result['design'] != design:
            raise ValueError('Stored decomposition candidates use a different design or input archive')
        validate(result, step1['rows'], len(chunks), decompositions)
        return result

    bm25 = run.BM25([run.chunk_doc(c) for c in chunks], K1, B)
    lexical = cached_lexical_ranks(database)
    archived = set(lexical)
    for row in step1['rows']:
        for text, ids in zip(row['texts'][1:], row['lists'][1:], strict=True):
            if text in lexical and lexical[text] != ids:
                raise ValueError(f'Archived lexical ranks disagree with Step 1 rule lists for question {row["qid"]}')
            lexical[text] = ids

    rows = []
    hits = computed = 0
    for frozen in step1['rows']:
        qid = frozen['qid']
        systems = {}
        for arm, raw in decompositions.items():
            started = perf_counter()
            text, split_seconds = raw[qid]
            splits = llm_compare.valid_lines(text)
            if splits:
                texts = [frozen['texts'][0], *splits]
                lists = [frozen['lists'][0]]
                for text in splits:
                    if text not in lexical:
                        lexical[text] = bm25.search(text, DEPTH)
                        computed += 1
                    hits += text in archived
                    lists.append(lexical[text])
            else:
                texts, lists = frozen['texts'], frozen['lists']
            systems[arm] = {'texts': texts, 'lists': lists, 'fallback': not bool(splits),
                            'candidate_seconds': perf_counter() - started, 'split_seconds': split_seconds}
        rows.append({'qid': qid, 'question_type': frozen['question_type'], 'systems': systems})
    result = {'design': design, 'rows': rows}
    validate(result, step1['rows'], len(chunks), decompositions)
    write_json(CANDIDATES, result)
    print(f'Built {computed:,} new lexical lists and reused {hits:,} archived lookups', flush=True)
    return result


if __name__ == '__main__':
    candidates = prepare_candidates()
    print(f"Saved decomposition candidates for {len(candidates['rows']):,} questions to {CANDIDATES}", flush=True)
