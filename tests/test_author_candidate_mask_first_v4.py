"""Synthetic-only checks; no real trajectory, model or tokenizer calls."""
import importlib.util
import hashlib
import math
import os
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("SYNSTEG_SOURCE_ROOT", ROOT))
PARSER_SHA = "cb17dd1780f5ee737e6b53454c0c7a911eb7867a9c2d2779f82790201d0b695f"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(ROOT / "src/synsteg_audit/author_candidate_mask_first_v4.py", "maskfirst")
p = SOURCE_ROOT / "src/synsteg_audit/author_candidate_schema_v3.py"
if hashlib.sha256(p.read_bytes()).hexdigest() != PARSER_SHA:
    raise RuntimeError("frozen parser identity differs")
parser = load(p, "frozen_schema")


def rows(special=None, default_piece=b"}", default_logit=0.0):
    special = special or {}
    result = []
    for token in range(50):
        piece, logit = special.get(token, (default_piece, default_logit))
        result.append({"token_id": token, "logit_hex": struct.pack("!f", logit).hex(), "piece_hex": piece.hex()})
    return sorted(result, key=lambda r: (-struct.unpack("!f", bytes.fromhex(r["logit_hex"]))[0], r["token_id"]))


class MaskFirstTests(unittest.TestCase):
    def test_tiny_legal_masses_survive_invalid_peak(self):
        result = m.filter_then_quantize(b'', ['word'], rows({48:(b' ', -100.0),49:(b'\t',-101.0)}), parser.classify_prefix)
        self.assertEqual(result['state'], 'ready')
        self.assertEqual(sum(r['weight'] for r in result['rows']), m.TOTAL)
        self.assertEqual(result['eligible_tokens'], 2)
        self.assertTrue(result['positive_support_structural_whitespace_only'])
        self.assertEqual(result['structural_whitespace_mass']['numerator'], m.TOTAL)
        self.assertIsNone(result['semantic_capacity_bits'])

    def test_empty_legal_set_has_no_fallback(self):
        result = m.filter_then_quantize(b'', ['word'], rows(), parser.classify_prefix)
        self.assertEqual(result['state'], 'empty_schema_support')
        self.assertEqual(len(result['rows']), 50)

    def test_budget_refuses_entire_probe_even_for_low_illegal_row(self):
        result = m.filter_then_quantize(b'', ['word'], rows({0:(b' ',1.0),49:(b'x'*20,-100.0)}), parser.classify_prefix,max_bytes=10)
        self.assertEqual(result['state'],'resource_refusal')
        self.assertEqual(len(result['rows']),50)

    def test_terminal_prefix_cannot_start_new_probe(self):
        with self.assertRaises(ValueError):
            m.filter_then_quantize(b'{"words":[{"word":[]}]}', ['word'], rows(), parser.classify_prefix)

    def test_single_legal_token_is_zero_choice_entropy(self):
        result=m.filter_then_quantize(b'', ['word'], rows({49:(b'{',-100.0)}), parser.classify_prefix)
        self.assertEqual(result['positive_tokens'],1)
        self.assertEqual(result['rows'][-1]['weight'],m.TOTAL)
        self.assertFalse(result['positive_support_structural_whitespace_only'])
        self.assertIsNone(result['semantic_capacity_bits'])

    def test_tie_breaks_by_token_id(self):
        result=m.filter_then_quantize(b'', ['word'], rows(default_piece=b' ',default_logit=1.0), parser.classify_prefix)
        self.assertEqual([r['weight'] for r in result['rows'][:16]],[m.TOTAL//50+1]*16)
        self.assertEqual([r['weight'] for r in result['rows'][16:]],[m.TOTAL//50]*34)

    def test_nonmax_underflow_is_explicit(self):
        result=m.filter_then_quantize(b'', ['word'], rows({0:(b'{',0.0),49:(b' ',-1e30)}), parser.classify_prefix)
        self.assertEqual(result['exponential_underflow_ids'],[49])
        self.assertEqual(result['positive_tokens'],1)

    def test_nonfinite_duplicate_unordered_and_incomplete_rows_rejected(self):
        cases=[]
        a=rows();a[0]['logit_hex']='7f800000';cases.append(a)
        a=rows();a[1]['token_id']=a[0]['token_id'];cases.append(a)
        a=rows();a.reverse();cases.append(a)
        a=rows()[:-1];cases.append(a)
        a=rows();a[0]['token_id']=False;cases.append(a)
        for value in cases:
            with self.subTest(value=value[0]):
                with self.assertRaises(ValueError):m.filter_then_quantize(b'', ['word'], value, parser.classify_prefix)

    def test_whitespace_outside_string_is_distinguished_from_inside(self):
        self.assertTrue(m.structural_whitespace(b'{"words": [',b' \t\n'))
        self.assertFalse(m.structural_whitespace(b'{"words": [{"word": ["two',b' '))
        self.assertFalse(m.structural_whitespace(b'{"words": [{"word": ["escaped\\\"',b' '))
        self.assertTrue(m.structural_whitespace(b'{"words": [{"word"',b' '))
        self.assertFalse(m.structural_whitespace(b'',b'\xc2\xa0'))

    def test_whitespace_streak_is_passive_and_nonprogress_can_exhaust_budget(self):
        raw=b'';streak=0
        for _ in range(512):
            self.assertEqual(parser.classify_prefix(raw,['word'])['state'],'incomplete')
            streak=m.whitespace_streak(raw,b' ',streak);raw+=b' '
        self.assertEqual(streak,512)
        self.assertEqual(parser.classify_prefix(raw,['word'])['state'],'incomplete')
        self.assertEqual(m.whitespace_streak(raw,b'{',streak),0)


if __name__ == '__main__':unittest.main()
