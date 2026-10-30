"""Scientific-integrity guards for the offline reproducibility pipeline."""
import copy
import importlib.util
import json
import hashlib
import zipfile
from unittest import mock
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'reproduce', Path(__file__).resolve().parents[1] / 'scripts/reproduce.py')
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'clean_detection').mkdir()
        base = {
            'setting': 'synthetic', 'seed': 0,
            'train_uuids': ['train-pair'], 'validation_uuids': ['validation-pair'],
            'test_uuids': ['test-pair'], 'errors': 0, 'accuracy': 1.0,
            'AUC': 1.0, 'confusion_matrix_01': [[1, 0], [0, 1]],
            'predictions': [
                {'uuid': 'test-pair', 'arm': 'cover', 'true': 0, 'pred': 0, 'p_stego': .1},
                {'uuid': 'test-pair', 'arm': 'stego', 'true': 1, 'pred': 1, 'p_stego': .9},
            ],
        }
        self.runs = [copy.deepcopy(base) for _ in range(96)]
        self.runs += [{'n_permutations': 50}, {'n_permutations': 50}]
        self.runs += [{'setting': 'length_only'} for _ in range(10)]

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        (self.root / 'clean_detection/results.json').write_text(json.dumps({'runs': self.runs}))
        (self.root / 'clean_detection/additional_controls_results.json').write_text(json.dumps({'runs': []}))
        return reproduce.validate_detection(self.root)

    def test_valid_pair_manifests_pass(self):
        result = self.validate()
        self.assertEqual(result['classifier_fits'], 206)
        self.assertEqual(result['manifest_runs_checked'], 96)

    def test_parent_document_leakage_fails(self):
        self.runs[0]['train_uuids'].append('test-pair')
        with self.assertRaisesRegex(ValueError, 'UUID split intersection'):
            self.validate()

    def test_incorrect_arm_label_fails(self):
        self.runs[0]['predictions'][0]['true'] = 1
        with self.assertRaisesRegex(ValueError, 'Incorrect stored label'):
            self.validate()

    def test_missing_pair_arm_fails(self):
        self.runs[0]['predictions'].pop()
        with self.assertRaisesRegex(ValueError, 'both arms'):
            self.validate()

    def test_saved_accuracy_must_match_predictions(self):
        self.runs[0]['accuracy'] = .5
        with self.assertRaisesRegex(ValueError, 'Metric does not match predictions'):
            self.validate()

    def test_invalid_probabilities_fail(self):
        for bad in (float('nan'), float('inf'), float('-inf'), -.01, 1.01):
            with self.subTest(probability=bad):
                self.runs[0]['predictions'][0]['p_stego'] = bad
                with self.assertRaisesRegex(ValueError, 'Invalid probability'):
                    self.validate()

    def test_saved_auc_must_match_predictions(self):
        self.runs[0]['AUC'] = .5
        with self.assertRaisesRegex(ValueError, 'AUC does not match'):
            self.validate()

    def test_saved_confusion_matrix_must_match_predictions(self):
        self.runs[0]['confusion_matrix_01'] = [[0, 1], [1, 0]]
        with self.assertRaisesRegex(ValueError, 'Confusion matrix does not match'):
            self.validate()

    def test_nonfinite_accuracy_and_auc_fail(self):
        for field in ('accuracy', 'AUC'):
            with self.subTest(metric=field):
                self.runs[0][field] = float('nan')
                with self.assertRaises(ValueError):
                    self.validate()
                self.runs[0][field] = 1.0

    def test_source_hash_tracks_data_but_ignores_bytecode(self):
        (self.root / 'source.json').write_text('one')
        first = reproduce.file_manifest(self.root)
        bytecode = self.root / '__pycache__'
        bytecode.mkdir()
        (bytecode / 'example.pyc').write_bytes(b'compiled')
        self.assertEqual(first, reproduce.file_manifest(self.root))
        (self.root / 'source.json').write_text('two')
        self.assertNotEqual(first, reproduce.file_manifest(self.root))


class SourceIdentityTests(unittest.TestCase):
    """Use a tiny ZIP fixture with the identity constants patched only in tests."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.archive = self.root / 'source.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('synsteg_reproduction/example.json', b'original')
        (self.source / 'example.json').write_bytes(b'original')
        self.hash_patch = mock.patch.object(reproduce, 'EXPECTED_ARCHIVE_SHA256',
                                           hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.count_patch = mock.patch.object(reproduce, 'EXPECTED_SOURCE_FILE_COUNT', 1)
        self.hash_patch.start()
        self.count_patch.start()

    def tearDown(self):
        self.count_patch.stop()
        self.hash_patch.stop()
        self.temp.cleanup()

    def test_exact_archive_members_pass(self):
        self.assertEqual(len(reproduce.verify_source_archive(self.source, self.archive)), 1)

    def test_modified_source_fails(self):
        (self.source / 'example.json').write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'changed='):
            reproduce.verify_source_archive(self.source, self.archive)

    def test_missing_source_member_fails(self):
        (self.source / 'example.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing='):
            reproduce.verify_source_archive(self.source, self.archive)

    def test_extra_source_member_fails(self):
        (self.source / 'extra.json').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'extra='):
            reproduce.verify_source_archive(self.source, self.archive)

    def test_wrong_archive_fails_even_if_extraction_matches(self):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('synsteg_reproduction/example.json', b'replacement')
        (self.source / 'example.json').write_bytes(b'replacement')
        with self.assertRaisesRegex(ValueError, 'archive hash mismatch'):
            reproduce.verify_source_archive(self.source, self.archive)

    def test_missing_archive_fails(self):
        self.archive.unlink()
        with self.assertRaisesRegex(ValueError, 'archive is required'):
            reproduce.verify_source_archive(self.source, self.archive)


if __name__ == '__main__':
    unittest.main()
