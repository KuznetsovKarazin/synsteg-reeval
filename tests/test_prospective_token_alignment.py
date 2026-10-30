"""Candidate alignment checks; fixed examples never use prospective samples."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'prospective_local_adapter_alignment', ROOT / 'scripts/prospective_local_adapter.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class FixedEncoding:
    def __init__(self, ids, offsets):
        self.ids, self.offsets = ids, offsets
        self.seen = []

    def __call__(self, text, **kwargs):
        self.seen.append(text)
        assert kwargs == {'add_special_tokens': False, 'return_offsets_mapping': True}
        return {'input_ids': self.ids, 'offset_mapping': self.offsets}


@pytest.mark.parametrize('ids,offsets', [
    ([99, 20, 3], [(0, 1), (1, 6), (6, 7)]),  # Retokenized causal prefix.
    ([1, 20, 21, 3], [(0, 1), (1, 4), (4, 6), (6, 7)]),  # Two BPE tokens.
    ([1, 20], [(0, 1), (1, 7)]),  # Token absorbs the punctuation suffix.
    ([1, 20, 3], [(0, 1), (3, 6), (6, 7)]),  # Core is only partially covered.
])
def test_unsafe_token_reuse_is_rejected(ids, offsets):
    tokenizer = FixedEncoding(ids, offsets)
    assert adapter.aligned_candidate_token(tokenizer, 'A boat.', 2, 6, 'ship',
        [1, 2, 3], [(0, 1), (1, 6), (6, 7)]) is None
    assert tokenizer.seen == ['A ship.']


def test_no_synthetic_bos_or_multitoken_candidate_is_introduced():
    tokenizer = FixedEncoding([], [])
    assert adapter.aligned_candidate_token(tokenizer, 'boat.', 0, 4, 'ship',
        [2, 3], [(0, 4), (4, 5)]) is None
    assert adapter.aligned_candidate_token(tokenizer, 'A boat.', 2, 6, 'two words',
        [1, 2, 3], [(0, 1), (1, 6), (6, 7)]) is None
    assert tokenizer.seen == []


@pytest.fixture(scope='module')
def pinned_gpt2_tokenizer():
    transformers = pytest.importorskip('transformers')
    try:
        return transformers.AutoTokenizer.from_pretrained('openai-community/gpt2',
            revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', local_files_only=True)
    except OSError:
        pytest.skip('Pinned GPT-2 tokenizer is not present in the local cache')


@pytest.mark.parametrize('cover,expected_index,expected_id', [
    ('A boat floats.', 1, 4074),
    ('A\nboat floats.', 2, 6720),
    ('A "boat" floats.', 2, 6720),
    ('A\tboat floats.', 2, 6720),
    ('A  boat floats.', 2, 4074),
    ('"boat" floats.', 1, 6720),
])
def test_pinned_tokenizer_uses_actual_separator(cover, expected_index, expected_id,
                                              pinned_gpt2_tokenizer):
    tokenizer = pinned_gpt2_tokenizer
    encoded = tokenizer(cover, add_special_tokens=False, return_offsets_mapping=True)
    start = cover.index('boat')
    assert adapter.aligned_candidate_token(tokenizer, cover, start, start + 4, 'ship',
        encoded['input_ids'], encoded['offset_mapping']) == (expected_index, expected_id)


def test_pinned_tokenizer_handles_changed_word_length_and_rejects_multi_bpe(
        pinned_gpt2_tokenizer):
    tokenizer = pinned_gpt2_tokenizer
    cover = 'A boat floats.'
    encoded = tokenizer(cover, add_special_tokens=False, return_offsets_mapping=True)
    result = adapter.aligned_candidate_token(tokenizer, cover, 2, 6, 'vessel',
        encoded['input_ids'], encoded['offset_mapping'])
    assert result == (1, tokenizer.encode(' vessel', add_special_tokens=False)[0])
    assert adapter.aligned_candidate_token(tokenizer, cover, 2, 6,
        'antidisestablishmentarianism', encoded['input_ids'],
        encoded['offset_mapping']) is None


def test_pinned_tokenizer_excludes_unconditioned_first_word(pinned_gpt2_tokenizer):
    tokenizer = pinned_gpt2_tokenizer
    encoded = tokenizer('boat floats.', add_special_tokens=False, return_offsets_mapping=True)
    assert adapter.aligned_candidate_token(tokenizer, 'boat floats.', 0, 4, 'ship',
        encoded['input_ids'], encoded['offset_mapping']) is None
