"""Independent receiver checks against fixed vectors, without codec imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cryptography.exceptions import InvalidTag

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_shared_core', ROOT / 'scripts/shared_dictionary_audit_core.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
VECTOR_PATH = ROOT / 'tests/fixtures/shared_dictionary_v1_vectors.json'


class IndependentSharedReceiver(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(VECTOR_PATH.read_text())
        self.vector = self.data['vectors'][0]
        self.key = bytes.fromhex(self.data['key_hex'])

    def test_both_fixed_reference_frames_decode_independently(self):
        result = audit.verify_reference(VECTOR_PATH)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual([v['frame_bits'] for v in result['vectors']], [160, 32])
        self.assertFalse(result['experimental_codec_imported'])

    def test_reference_frame_mutation_is_rejected(self):
        self.data['vectors'][0]['frame_hex'] = '00'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(self.data))
            with self.assertRaisesRegex(ValueError, 'frame_hex'):
                audit.verify_reference(path)

    def test_distinct_payload_label_change_fails_authentication(self):
        text = self.vector['stego_text']
        analysis = audit.analyze(text, self.data['dictionary'], self.vector['protocol'])
        _, mapping = audit.derive_keys(self.key, analysis['dictionary_sha256'], analysis['protocol_sha256'])
        slot = analysis['slots'][20]
        groups = audit.groups(analysis, slot, mapping)
        alternative = next(word for group in groups if slot['surface'].lower() not in group for word in group)
        changed = text[:slot['start']] + alternative + text[slot['end']:]
        with self.assertRaises(InvalidTag):
            audit.decode(changed, self.key, self.data['dictionary'], self.vector['protocol'])

    def test_same_label_change_preserves_authenticated_message(self):
        text = self.vector['stego_text']
        analysis = audit.analyze(text, self.data['dictionary'], self.vector['protocol'])
        _, mapping = audit.derive_keys(self.key, analysis['dictionary_sha256'], analysis['protocol_sha256'])
        slot = analysis['slots'][20]
        group = next(group for group in audit.groups(analysis, slot, mapping) if slot['surface'].lower() in group)
        alternative = next(word for word in group if word != slot['surface'].lower())
        changed = text[:slot['start']] + alternative + text[slot['end']:]
        self.assertEqual(audit.decode(changed, self.key, self.data['dictionary'], self.vector['protocol'])['payload_hex'], self.data['payload_hex'])

    def test_literal_change_or_wrong_key_reject(self):
        for text, key in [(self.vector['stego_text'].replace('Public', 'PUBLIC', 1), self.key),
                          (self.vector['stego_text'], self.key[::-1])]:
            with self.subTest(text_start=text[:25]), self.assertRaises((ValueError, InvalidTag)):
                audit.decode(text, key, self.data['dictionary'], self.vector['protocol'])

    def test_public_case_and_punctuation_rules(self):
        text = 'Small, SMALL! small? sMaLl small-fast small2 _small яsmall «small» small\u0301'
        analysis = audit.analyze(text, self.data['dictionary'], self.vector['protocol'])
        self.assertEqual([s['surface'] for s in analysis['slots']], ['Small', 'SMALL', 'small', 'small'])
        self.assertEqual([s['case_style'] for s in analysis['slots']], ['title', 'upper', 'lower', 'lower'])
        self.assertEqual([s['token_index'] for s in analysis['slots']], [0, 1, 2, 8])

    def test_single_letter_dictionary_and_unbound_protocol_fields_reject(self):
        dictionary = copy.deepcopy(self.data['dictionary'])
        dictionary['classes'][0]['words'].append('a')
        with self.assertRaises(ValueError):
            audit.dictionary_identity(dictionary)
        for change in [{'version': 'different'}, {'cover': 'hidden side state'}, {'min_group_size': True}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.protocol_identity(self.vector['protocol'] | change)


if __name__ == '__main__':
    unittest.main()
