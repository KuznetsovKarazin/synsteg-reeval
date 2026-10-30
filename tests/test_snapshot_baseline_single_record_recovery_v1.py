import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).parent
HELPER_PATH = HERE.parent / 'scripts' / 'snapshot_baseline_single_record_recovery_v1.py'
if not HELPER_PATH.is_file():
    HELPER_PATH = HERE / 'snapshot_baseline_single_record_recovery_v1.py'
s = importlib.util.spec_from_file_location('snapshot_under_test', HELPER_PATH)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def fixture(n=5):
    prior = {str(i): {'status': 'completed', 'result': {'usage': {'output_tokens': 10}, 'stop_reason': 'end_turn'}}
             for i in range(1, 82)}
    prior['81']['result'] = {'usage': {'output_tokens': 1024}, 'stop_reason': 'max_tokens'}
    prior['82'] = {'status': 'reserved_outcome_unknown', 'result': None}
    new = [{'status': 'completed', 'result': {'usage': {'output_tokens': 100}, 'stop_reason': 'end_turn'}} for _ in range(n)]
    records = []
    for model in ('qwen05', 'tinyllama11', 'falcon3_1b'):
        for i in range(1, 101):
            status = 'not_attempted'
            if model == 'tinyllama11':
                if i <= 81 or 83 <= i < 83+n: status = 'completed'
                elif i == 82: status = 'reserved_outcome_unknown'
            records.append({'model_alias': model, 'source_order': i, 'status': status})
    projection = {'schema': m.ID + '-projection', 'records': records,
                  'counts': {'completed':81+n, 'reserved_outcome_unknown':1, 'not_attempted':218-n},
                  'full_denominator':300, 'tinyllama_denominator':100}
    return projection, prior, new, list(range(83,83+n))


class SnapshotChecks(unittest.TestCase):
    def test_complete_prefix_totals_and_full_denominator(self):
        p,old,new,orders=fixture(); before=copy.deepcopy(p)
        x=m.summary_from_inputs(p,old,new,orders)
        self.assertEqual(x['counts'],{'completed':86,'reserved_outcome_unknown':1,'not_attempted':213})
        self.assertEqual(x['completed_output_tokens'],2324)
        self.assertEqual(x['stop_reasons'],{'end_turn':85,'max_tokens':1})
        self.assertEqual(p,before)

    def test_terminal_100_still_not_100_tiny_successes(self):
        x=m.summary_from_inputs(*fixture(18))
        self.assertEqual(x['counts'],{'completed':99,'reserved_outcome_unknown':1,'not_attempted':200})

    def test_unknown_not_relabelled_or_counted_as_tokens(self):
        p,old,new,orders=fixture();old['82']['result']={'usage':{'output_tokens':999999}}
        with self.assertRaisesRegex(ValueError,'Historical81'):m.summary_from_inputs(p,old,new,orders)

    def test_new_failure_not_silently_dropped(self):
        p,old,new,orders=fixture();new[-1]['status']='reserved_outcome_unknown'
        with self.assertRaisesRegex(ValueError,'complete clean prefix'):m.summary_from_inputs(p,old,new,orders)

    def test_denominator_disagreement_refused(self):
        p,old,new,orders=fixture();p['counts']['completed']+=1
        with self.assertRaisesRegex(ValueError,'projection/count mismatch'):m.summary_from_inputs(p,old,new,orders)

    def test_destination_is_one_shot(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'new';m.write_snapshot(dest,{'summary.json':{'completed':86}})
            before=(dest/'summary.json').read_bytes()
            with self.assertRaises(FileExistsError):m.write_snapshot(dest,{'summary.json':{'completed':999}})
            self.assertEqual((dest/'summary.json').read_bytes(),before)

    def test_busy_lease_and_wrong_helper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lease=root/'lease';lease.write_bytes(b'')
            with m.existing_lease(lease):
                with self.assertRaises(BlockingIOError):
                    with m.existing_lease(lease): pass
            script=root/m.HELPER;script.parent.mkdir();script.write_text('raise AssertionError("must not import")\n')
            with self.assertRaisesRegex(ValueError,'Frozen helper'):m.frozen_helper(root)


if __name__ == '__main__':
    unittest.main()
