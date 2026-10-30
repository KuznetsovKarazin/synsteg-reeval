"""Pinned local-model adapter for a small prospective feasibility experiment.

WordNet first-sense/POS filtering is a conservative heuristic, not a semantic
equivalence guarantee. GPT-2 ranks single-BPE candidates in the original cover
context, before payload creation; it never queries an attack classifier.
Each admitted replacement is tokenized in its exact rendered context, with
the causal prefix checked before any original-cover logits are reused.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit.fixed_codebook import freeze_codebook, FrozenCodebook, encode_bits, decode_bits


def aligned_candidate_token(tokenizer, cover, start, end, candidate,
                            original_ids, original_offsets):
    """Return a verified (causal token index, candidate token ID), or None.

    Whitespace and punctuation are retained exactly. The candidate's whole
    lexical core must occupy one token, with no suffix in that token, and the
    preceding token IDs must equal the original cover's prefix. These guards
    make reuse of the original causal logits valid. A first token is excluded
    because the protocol supplies no BOS conditioning token.
    """
    if not 0 <= start < end <= len(cover):
        raise ValueError('Invalid candidate character span')
    if re.fullmatch('[a-z]+', candidate) is None:
        return None
    original_index = next((j for j, (a, b) in enumerate(original_offsets)
                           if a <= start < b), None)
    if original_index is None or original_index == 0:
        return None
    rendered = cover[:start] + candidate + cover[end:]
    encoded = tokenizer(rendered, add_special_tokens=False,
                        return_offsets_mapping=True)
    candidate_end = start + len(candidate)
    intersecting = [j for j, (a, b) in enumerate(encoded['offset_mapping'])
                    if a < candidate_end and b > start]
    if len(intersecting) != 1:
        return None
    index = intersecting[0]
    a, b = encoded['offset_mapping'][index]
    if (index != original_index or a > start or b != candidate_end
            or list(encoded['input_ids'][:index]) != list(original_ids[:index])):
        return None
    return index, encoded['input_ids'][index]


class LocalAdapter:
    def __init__(self):
        self.generator = self.generator_tokenizer = None
        self.scorer = self.scorer_tokenizer = None

    def generate_cover(self, sample, generation):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = generation['model']
        torch.set_num_threads(int(generation.get('cpu_threads', 2)))
        torch.manual_seed(sample['generation_seed'])
        torch.use_deterministic_algorithms(True)
        if self.generator is None:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(model['model_id'], revision=model['model_revision'], local_files_only=True)
            self.generator = AutoModelForCausalLM.from_pretrained(model['model_id'], revision=model['model_revision'], local_files_only=True, torch_dtype=torch.bfloat16).eval()
        torch.manual_seed(sample['generation_seed'])
        prompt = self.generator_tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'Write clear English prose. Return only the requested text, without a heading or commentary.'},
            {'role': 'user', 'content': sample['prompt']}], tokenize=False, add_generation_prompt=True)
        inputs = self.generator_tokenizer(prompt, return_tensors='pt')
        start = time.monotonic()
        with torch.inference_mode():
            output = self.generator.generate(**inputs, max_new_tokens=int(generation['max_new_tokens']),
                do_sample=True, temperature=float(generation['temperature']), top_p=float(generation['top_p']),
                top_k=int(generation.get('top_k', 50)), repetition_penalty=1.0,
                pad_token_id=self.generator_tokenizer.eos_token_id)
        ids = output[0, inputs['input_ids'].shape[1]:]
        return {'text': self.generator_tokenizer.decode(ids, skip_special_tokens=True).strip(),
                'metadata': {'model': model, 'dtype': 'bfloat16', 'device': 'cpu', 'seed': sample['generation_seed'],
                             'generated_tokens': len(ids), 'input_tokens': inputs['input_ids'].shape[1],
                             'hit_token_limit': len(ids) == generation['max_new_tokens'],
                             'seconds': time.monotonic() - start}}

    def prepare_methods(self, cover, sample, method_config):
        import torch
        import nltk
        from nltk.corpus import wordnet as wn
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(int(method_config.get('cpu_threads', 2)))
        if self.scorer is None:
            model = method_config['quality_model']
            self.scorer_tokenizer = AutoTokenizer.from_pretrained(model['model_id'], revision=model['revision'], local_files_only=True)
            self.scorer = AutoModelForCausalLM.from_pretrained(model['model_id'], revision=model['revision'], local_files_only=True, torch_dtype=torch.float32).eval()
        spans = list(re.finditer(r'\S+', cover))
        cores = []
        word_starts = []
        for span in spans:
            match = re.fullmatch(r'([^A-Za-z0-9_]*)([A-Za-z]+)([^A-Za-z0-9_]*)', span.group())
            cores.append(match.group(2) if match else span.group())
            word_starts.append(span.start() + len(match.group(1)) if match else None)
        tags = nltk.pos_tag(cores, lang='eng')
        pos_map = {'NN': wn.NOUN, 'JJ': wn.ADJ, 'RB': wn.ADV, 'VB': wn.VERB}
        candidates = []
        for word, tag in tags:
            pos = pos_map.get(tag)
            options = []
            if pos and word.islower() and word.isalpha() and (wn.morphy(word, pos) or word) == word:
                synsets = wn.synsets(word, pos=pos)
                if synsets:
                    options = [lemma.name() for lemma in synsets[0].lemmas() if re.fullmatch('[a-z]+', lemma.name())]
            candidates.append(options)
        encoded = self.scorer_tokenizer(cover, add_special_tokens=False,
            return_tensors='pt', return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        original_ids = encoded['input_ids'][0].tolist()
        if encoded['input_ids'].shape[1] > 1024:
            raise ValueError('Pilot cover exceeds frozen quality-model context; no silent truncation')
        with torch.inference_mode():
            logits = self.scorer(**encoded).logits[0]
        probabilities = {}
        alignment = {'candidate_pairs_considered': 0, 'candidate_pairs_admitted': 0,
                     'candidate_pairs_rejected': 0}
        for i, (word, start) in enumerate(zip(cores, word_starts)):
            if start is None or not candidates[i]:
                continue
            scores = None
            for candidate in sorted(set([word, *candidates[i]])):
                alignment['candidate_pairs_considered'] += 1
                token = aligned_candidate_token(self.scorer_tokenizer, cover,
                    start, start + len(word), candidate, original_ids, offsets)
                if token is None:
                    alignment['candidate_pairs_rejected'] += 1
                    continue
                index, token_id = token
                if scores is None:
                    scores = torch.log_softmax(logits[index-1].double(), dim=-1)
                probabilities[i, candidate] = -float(scores[token_id])
                alignment['candidate_pairs_admitted'] += 1
        del logits
        def allowed(context):
            return (context.token_index, context.candidate) in probabilities
        def quality(context):
            return probabilities[context.token_index, context.candidate]
        book = freeze_codebook(cover, candidates, seed='prospective-fixed-v1:' + sample['id'],
            max_bits_per_position=1, min_group_size=1, protect_capitalized=True,
            candidate_filter=allowed, candidate_filter_id='wordnet-first-sense-base-pos-rendered-single-gpt2-token-v2',
            scorer=quality, scorer_id='gpt2-verified-cover-prefix-next-token-nll-v2')
        state = {'codebook': book.to_dict(), 'codebook_sha256': book.sha256,
                 'eligibility': book.eligibility, 'capacity_bits': book.capacity_bits,
                 'framing': 'external_length', 'external_length_bits': 32,
                 'token_alignment': alignment,
                 'semantic_equivalence_verified': False}
        return {name: state for name in method_config['variants']}

    def encode_decode(self, variant, prepared, payload_bits, cover, sample, method_config):
        book = FrozenCodebook.from_dict(prepared['codebook'])
        if book.sha256 != prepared['codebook_sha256']:
            raise ValueError('Prepared codebook changed')
        selection = {'fixed_canonical': 'canonical', 'fixed_quality': 'quality'}[variant]
        result = encode_bits(book, payload_bits, selection=selection, framing='external_length')
        recovered = decode_bits(book, result.text, framing='external_length', expected_bits=len(payload_bits))
        return {'stego_text': result.text, 'decoded_payload': recovered, 'embedded_bits': len(payload_bits),
                'key': book.to_dict(), 'metadata': {**asdict(result), 'text': None,
                'external_length_bits': 32, 'eligibility': book.eligibility, 'capacity_bits': book.capacity_bits,
                'semantic_equivalence_verified': False}}


def make_adapter():
    return LocalAdapter()
