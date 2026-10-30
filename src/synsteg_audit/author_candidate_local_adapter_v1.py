"""Offline boundary for a new local-provider author-candidate comparator.

No API, model, tokenizer or generation imports. The original ZIP is read-only.
Only hash-verified, named pure author functions are compiled; API modules and
module-level code are never imported. This is not historical provider replication.
"""
from __future__ import annotations
import ast
import hashlib
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

ID = 'author-candidate-local-adapter-v1'
ARCHIVE_SHA256 = '793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790'
MEMBERS = {
    'steganography/secret_key.py': 'b0eb16096213027865b843009cd4f46e6fba73a132b5a0000c42d91b6383ebe6',
    'steganography/helper.py': '0c4bf88f77aeaa705d5a99611e63898bf2b23beac1f566d4212b502188b3fef9',
    'utils/prompts.py': '173441562eb914d1ac4ff3aae178866e64d5e3d83f0b1b87dc6f1076799e405e',
}
PURE = {
    'steganography/helper.py': ('clean_container', 'remove_brackets'),
    'steganography/secret_key.py': ('binarize_synonyms', 'clean_secret_key', 'align_container_and_secret_key'),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_author(archive):
    """Read only selected source members; no imports or provider calls."""
    archive = Path(archive)
    if digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError('author archive differs')
    namespace = {'product': itertools.product, 'MIN_DISTINCT_SHARE': 0.9,
                 'PUNCTUATION': ',.?!', 'BRACKETS': '[](){}'}
    prompts = None
    with ZipFile(archive) as z:
        sources = {}
        for member, expected in MEMBERS.items():
            raw = z.read('synsteg_reproduction/' + member)
            if digest(raw) != expected:
                raise ValueError('author source differs: ' + member)
            sources[member] = raw.decode('utf-8')
        # Hash-pinned author pure functions, without unrelated statements/imports.
        for member, names in PURE.items():
            tree = ast.parse(sources[member], filename=member)
            selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            if {n.name for n in selected} != set(names):
                raise ValueError('pure source function set differs')
            exec(compile(ast.Module(body=selected, type_ignores=[]), member, 'exec'), namespace)
        tree = ast.parse(sources['utils/prompts.py'])
        prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    return namespace, prompts


def make_plan(archive, cover, *, bits_per_word=3):
    if type(bits_per_word) is not int or not 1 <= bits_per_word <= 5:
        raise ValueError('bits_per_word must be integer1..5')
    if not isinstance(cover, str) or not cover.split():
        raise ValueError('nonempty cover required')
    _, prompts = load_author(archive)
    words = cover.split()
    system = prompts['ALL_SYNONYMS_GENERATION_PROMPT'].replace('N_SYNONYMS', str(2 ** bits_per_word))
    requests = []
    for order, start in enumerate(range(0, len(words), 5)):
        chunk = words[start:start + 5]
        user = prompts['ALL_SYNONYMS_GENERATION_INPUT'].format(context=chunk)
        record = {'chunk_order': order, 'word_start': start, 'words': chunk,
                  'system': system, 'user': user, 'temperature_argument': 0.7,
                  'json_shape_attempt_limit': 3}
        requests.append({**record, 'request_sha256': digest(canonical(record))})
    return {'schema': ID + '-request-plan', 'archive_sha256': ARCHIVE_SHA256,
            'author_source_sha256': MEMBERS, 'cover_text': cover,
            'cover_sha256': digest(cover.encode()), 'bits_per_word': bits_per_word,
            'additional_bits': 0, 'chunk_words': 5, 'requested_chunks': len(requests),
            'requested_words': len(words), 'requests': requests,
            'provider': 'unbound; separate frozen local native activation required',
            'generation_enabled': False, 'payload_input': False}


def parse_response(raw_text):
    """New strict shape gate; intentionally stricter than author's first-dict test."""
    if not isinstance(raw_text, str):
        raise ValueError('raw response must be a string')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    value = json.loads(raw_text, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(value, dict) or set(value) != {'words'}:
        raise ValueError('expected only words field')
    words = value['words']
    if not isinstance(words, list) or not words:
        raise ValueError('nonempty words list required')
    for row in words:
        if not isinstance(row, dict) or len(row) != 1:
            raise ValueError('each entry must have one base token')
        key, candidates = next(iter(row.items()))
        if (not isinstance(key, str) or not key.strip() or not isinstance(candidates, list)
                or any(not isinstance(x, str) or not x.strip() for x in candidates)):
            raise ValueError('invalid base or candidate string list')
    return words


def resolve_replies(request, replies):
    """Interpret preserved replies, never invoke a provider or retry UNKNOWN.

    replies is the ordered evidence supplied by an external coordinator. No usage
    is fabricated. Missing replies are not attempted, not zero-capacity results.
    """
    if not isinstance(replies, list) or len(replies) > 3:
        raise ValueError('at most3 ordered shape attempts')
    accepted = None
    outcome = 'not_attempted'
    evidence = []
    for index, reply in enumerate(replies):
        if outcome in {'accepted', 'unknown'}:
            raise ValueError('attempt after terminal accepted/unknown reply')
        if not isinstance(reply, dict) or set(reply) - {'state', 'raw_text', 'usage', 'detail'}:
            raise ValueError('invalid reply evidence fields')
        if reply.get('state') not in {'returned', 'unknown'}:
            raise ValueError('invalid reply state')
        item = {'attempt_order': index, **reply}
        if reply['state'] == 'unknown':
            if 'raw_text' in reply:
                raise ValueError('unknown reply cannot have asserted returned text')
            outcome = 'unknown'
        else:
            raw = reply.get('raw_text')
            if not isinstance(raw, str):
                raise ValueError('returned reply needs exact raw text')
            item['raw_sha256'] = digest(raw.encode())
            try:
                accepted = parse_response(raw)
                outcome = 'accepted'
            except (ValueError, TypeError, RecursionError) as exc:
                outcome = 'shape_rejected'
                item['rejection'] = type(exc).__name__ + ': ' + str(exc)
        item['outcome'] = outcome
        evidence.append(item)
    if outcome == 'shape_rejected' and len(replies) == 3:
        outcome = 'shape_attempts_exhausted'
    return {'request_sha256': request['request_sha256'], 'chunk_order': request['chunk_order'],
            'outcome': outcome, 'attempts': evidence, 'accepted_words': accepted}


def assemble_pool(archive, plan, replies_by_chunk):
    """Build original a=0 pool before any payload; require all chunks accepted."""
    expected = make_plan(archive, plan.get('cover_text'), bits_per_word=plan.get('bits_per_word'))
    if canonical(plan) != canonical(expected):
        raise ValueError('plan altered or contains unbound fields')
    if not isinstance(replies_by_chunk, list) or len(replies_by_chunk) != len(plan['requests']):
        raise ValueError('full ordered chunk denominator required')
    outcomes = [resolve_replies(req, replies) for req, replies in zip(plan['requests'], replies_by_chunk)]
    result = {'schema': ID + '-candidate-pool', 'plan_sha256': digest(canonical(plan)),
              'requested_chunks': len(outcomes), 'chunk_outcomes': outcomes,
              'accepted_chunks': sum(r['outcome'] == 'accepted' for r in outcomes),
              'raw_pre_payload_words': None, 'binarized_pre_payload_pool': None,
              'aligned_pre_payload_pool': None, 'aligned_pool_sha256': None,
              'status': 'candidate_resource_incomplete', 'payload_seen': False,
              'inference_calls_in_this_function': 0, 'encode_decode_calls': 0,
              'semantic_equivalence_verified': False}
    if result['accepted_chunks'] != len(outcomes):
        return result
    raw = [row for out in outcomes for row in out['accepted_words']]
    source, _ = load_author(archive)
    binarized = [{base: source['binarize_synonyms'](base, options, plan['bits_per_word'])}
                 for row in raw for base, options in row.items()]
    aligned = source['align_container_and_secret_key'](plan['cover_text'], binarized)
    result.update(status='complete_pre_payload_pool', raw_pre_payload_words=raw,
                  binarized_pre_payload_pool=binarized, aligned_pre_payload_pool=aligned,
                  aligned_pool_sha256=digest(canonical(aligned)))
    return result


def save_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
