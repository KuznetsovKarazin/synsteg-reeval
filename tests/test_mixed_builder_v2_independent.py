"""Independent additive-builder regressions; every new outcome is synthetic."""
import copy
import importlib.util
from pathlib import Path
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('builder_v2_author_fixture', ROOT / 'tests/test_mixed_development_snapshot_builder_v2.py')
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
b, m = author.b, author.m


class IndependentBuilderTests(unittest.TestCase):
    def setUp(self):
        self.case = author.SuccessorBuilderTests(methodName='test_first_plus_source2_with_shared_full100_contract')
        self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.f, self.root = self.case.f, self.case.f.root

    def test_sparse_mixed_outcomes_preserve_all300_and_every_unchanged_row(self):
        self.case.qwen()
        prefix, identity = self.case.add_successor(2, 'reservation_only')
        self.f.write(prefix + '/process.json', {'identity': identity, 'source_order': 2, 'timeout_seconds': 1200,
            'automatic_retries': 0, 'status': 'stopped_requires_review_no_retry', 'timed_out': False})
        self.f.refresh(prefix)
        self.case.add_successor(3, 'failed'); self.case.add_successor(100)
        original = self.f.read(m.BASE_FILES['base'])
        result = self.case.build(self.case.request())
        self.assertEqual(result['counts'], {'completed': 91, 'reserved_outcome_unknown': 2, 'failed': 1, 'not_attempted': 206})
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 2, 'reserved_outcome_unknown': 1, 'failed': 1, 'not_attempted': 96})
        before = {(r['model_alias'], r['source_order']): r for r in original['records']}
        after = {(r['model_alias'], r['source_order']): r for r in result['records']}
        self.assertEqual(len(after), 300)
        changed = {('qwen05', n) for n in (1, 2, 3, 100)}
        for key in set(before) - changed:
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after[('tinyllama11', 82)]['status'], 'reserved_outcome_unknown')
        self.assertEqual(after[('tinyllama11', 91)]['status'], 'not_attempted')

    def test_wrong_directory_order_cannot_relabel_completed_result(self):
        self.case.qwen(); prefix, _ = self.case.add_successor(2)
        renamed = b.QWEN_SEQUENTIAL_ROOT + '/order003'
        shutil.move(self.root / prefix, self.root / renamed)
        with self.assertRaises(ValueError):
            self.case.build(self.case.request())

    def test_hidden_successor_cannot_be_removed_from_both_caller_indices(self):
        self.case.qwen(); prefix, _ = self.case.add_successor(2)
        request = self.case.request()
        del request['primary_receipts'][prefix]
        request['remote_census']['files_sha256'] = {p: digest for p, digest in request['remote_census']['files_sha256'].items()
                                                   if not p.startswith(prefix + '/')}
        with self.assertRaisesRegex(ValueError, 'local/remote'):
            b.stage(m, self.root, request)

    def test_full100_context_is_checked_even_for_first_plus_single_successor(self):
        self.case.qwen(); self.case.add_successor(2)
        path = b.QWEN_PRE + '/all100-context.json'
        context = self.f.read(path)
        context['records'][99]['input_tokens'] = 100000
        self.f.write(path, context)
        # Recompute caller binding too: semantic context validation must reject.
        with self.assertRaises(ValueError):
            self.case.build(self.case.request())

    def test_nonboolean_remote_verification_is_not_authorization(self):
        self.case.qwen(); self.case.add_successor(2)
        original = self.case.request()
        for field in ('readback_verified', 'branches_and_prs_checked'):
            with self.subTest(field=field):
                request = copy.deepcopy(original); request['remote_census'][field] = 1
                with self.assertRaises(ValueError):
                    b.stage(m, self.root, request)

    def test_symlink_runs_ancestor_and_existing_output_fail_closed(self):
        self.case.qwen(); self.case.add_successor(2); request = self.case.request()
        output = Path(self.case.tmp.name) / 'existing'
        output.mkdir(); (output / 'sentinel').write_text('unchanged')
        raw = b.encoded(request)
        with self.assertRaises(ValueError):
            b.build(m, self.root, raw, m.sha(raw), output)
        self.assertEqual((output / 'sentinel').read_text(), 'unchanged')
        outside = Path(self.case.tmp.name) / 'actual-runs'
        shutil.move(self.root / 'runs', outside)
        (self.root / 'runs').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            b.census(m, self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
