import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_recovery83_100_text_review_inputs_v1.py'
spec = importlib.util.spec_from_file_location('extract', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReviewInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.inputs = self.base / 'inputs'
        self.run = self.base / 'run'
        self.jobs = []
        for order in range(83, 101):
            row = {'record_id': f'id{order}', 'split': 'development', 'source_group_id': f'g{order}',
                   'title': f'Source{order}', 'excerpt': 'Source fact dated 1990.', 'prompt': f'prompt{order}',
                   'excerpt_sha256': m.sha_bytes(b'Source fact dated 1990.'),
                   'target_words': 500, 'genre': 'encyclopedic', 'topic': 'history'}
            path = f'prompts/order{order:03d}.jsonl'
            self.write(self.inputs / path, row)
            self.jobs.append({'source_order': order, 'record_id': f'id{order}', 'identity': f'identity{order}',
                              'prompt_path': path, 'prompt_file_sha256': m.sha_bytes((self.inputs / path).read_bytes()),
                              'original_prompt_sha256': m.sha_bytes(row['prompt'].encode())})
        self.protocol = self.inputs / 'protocol.json'
        self.write(self.protocol, {'experiment_id': m.RECOVERY, 'new_source_orders': list(range(83, 101)),
                                  'permanently_unknown_source_orders': [82], 'jobs': self.jobs})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')

    def process(self):
        path = self.run / 'order083/process.json'
        self.write(path, {'schema': m.RECOVERY + '-process', 'source_order': 83,
                          'status': 'completed_pending_remote_preservation'})
        return path

    def saved(self):
        directory = self.run / 'order083'
        process = self.process()
        result = {'cover_text': 'A source fact in 1990.\n\nA new block in 2001.', 'returned_model': m.MODEL,
                  'usage': {'input_tokens': 200, 'output_tokens': 20}, 'stop_reason': 'end_turn', 'hit_output_limit': False}
        entry = {'status': 'completed', 'result': result, 'metadata': {'record': {'record_id': 'id83'}}}
        self.write(directory / 'run/local_ledger/ledger.json', {'attempts': {'identity83': entry}})
        self.write(directory / 'run/attempts/identity83.json', {'status': 'completed', 'result': result, 'record': {'record_id': 'id83'}})
        receipt = {'readback_verified': True, 'commit': 'a' * 40, 'process_sha256': m.sha_bytes(process.read_bytes()),
                   'files_sha256': m.inventory(directory)}
        self.write(directory / 'git-preservation-receipt.json', receipt)
        return directory

    def test_full18_and_all_null_initial(self):
        report = m.build(self.protocol, self.run)
        self.assertEqual(report['availability_counts'], {'not_attempted': 18})
        self.assertEqual(len(report['records']), 18)
        self.assertEqual(report['campaign_denominator'], 300)
        self.assertEqual(report['historical_unknown_source_orders_not_retried'], [82])
        self.assertTrue(all(row['quality_labels'] is None and row['cover_text'] is None for row in report['records']))

    def test_live_poison_outputs_are_not_read(self):
        path = self.run / 'order083/run/local_ledger/ledger.json'
        path.parent.mkdir(parents=True)
        path.write_text('POISON_NOT_JSON')
        report = m.build(self.protocol, self.run)
        self.assertEqual(report['records'][0]['availability'], 'not_terminal_outputs_not_read')
        self.assertIsNone(report['records'][0]['cover_text'])

    def test_terminal_unpreserved_poison_not_read(self):
        self.process()
        path = self.run / 'order083/run/local_ledger/ledger.json'
        path.parent.mkdir(parents=True)
        path.write_text('POISON_NOT_JSON')
        report = m.build(self.protocol, self.run)
        self.assertEqual(report['records'][0]['availability'], 'terminal_pending_git_preservation_outputs_not_read')

    def test_saved_cover_exact_spans_and_null_quality(self):
        self.saved()
        report = m.build(self.protocol, self.run)
        row = report['records'][0]
        self.assertEqual(report['availability_counts'], {'not_attempted': 17, 'saved_completed_cover': 1})
        self.assertEqual([s['text'] for s in row['candidate_number_spans']], ['1990', '2001'])
        self.assertEqual(len(row['cover_blocks']), 2)
        for span in row['cover_blocks'] + row['candidate_number_spans']:
            self.assertEqual(row['cover_text'][span['start']:span['end']], span['text'])
        self.assertIsNone(row['quality_labels'])
        self.assertFalse(row['review_complete'])

    def test_changed_raw_evidence_rejected(self):
        directory = self.saved()
        path = directory / 'run/attempts/identity83.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            m.build(self.protocol, self.run)

    def test_terminal_unknown_retains_null(self):
        directory = self.saved()
        for path in [directory / 'run/local_ledger/ledger.json', directory / 'run/attempts/identity83.json']:
            value = json.loads(path.read_text())
            obj = value['attempts']['identity83'] if 'attempts' in value else value
            obj.update(status='reserved_outcome_unknown', result=None)
            self.write(path, value)
        receipt = json.loads((directory / 'git-preservation-receipt.json').read_text())
        receipt['files_sha256'] = m.inventory(directory)
        self.write(directory / 'git-preservation-receipt.json', receipt)
        row = m.build(self.protocol, self.run)['records'][0]
        self.assertEqual(row['generation_status'], 'reserved_outcome_unknown')
        self.assertIsNone(row['quality_labels'])
        self.assertIsNone(row['cover_text'])

    def test_source_mutation_and_symlink_refused(self):
        path = self.inputs / 'prompts/order100.jsonl'
        before = path.read_bytes()
        path.write_bytes(before + b' ')
        with self.assertRaisesRegex(ValueError, 'Input hash changed'):
            m.build(self.protocol, self.run)
        path.write_bytes(before)
        self.run.mkdir()
        (self.run / 'order083').symlink_to(self.inputs, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            m.build(self.protocol, self.run)


if __name__ == '__main__':
    unittest.main()
