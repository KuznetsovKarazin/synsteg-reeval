"""New diagnostic workflow guard tests; no production covers are analyzed."""
import copy
import hashlib
import importlib.util
import pathlib
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'scripts/measure_recovery83_100_capacity_v1.py'
spec = importlib.util.spec_from_file_location('capacity_new_workflow', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def snapshot():
    records = [dict(model_alias=model, source_order=n,
                    status='completed' if model == 'tinyllama11' and 83 <= n <= 90 else 'not_attempted')
               for model in ['qwen05','tinyllama11','falcon3_1b'] for n in range(1,101)]
    return dict(records=records, counts={'completed':8,'not_attempted':292})


def input_fixture(root):
    paths = [m.CORE, m.PUBLIC_PROTOCOL, m.SNAPSHOT,
             'data/shared_dictionary_recovery_v1.json', 'data/shared_dictionary_recovery_v2.json']
    for order in m.OBSERVED:
        paths.extend([f'experiments/baseline-single-record-recovery-v1/inputs/prompts/order{order:03d}.jsonl',
                      f'runs/baseline-single-record-recovery-v1/order{order:03d}/run/attempts/{order:064x}.json'])
    files = []
    for path in paths:
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'{}')
        files.append(dict(path=path,bytes=2,sha256=m.sha(b'{}'),git_blob_sha1=hashlib.sha1(b'blob 2\0{}').hexdigest()))
    return dict(commit=m.COMMIT,actual_git_readback=True,files=files)


class DiagnosticGuards(unittest.TestCase):
    def test_missing_is_not_zero(self):
        rows = m.frame_rows(91,'v2',None)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(x['capacity_bits'] is None and x['capacity_feasible'] is None for x in rows))

    def test_zero_is_observed_not_missing(self):
        rows = m.frame_rows(83,'v2',0)
        self.assertTrue(all(x['capacity_feasible'] is False for x in rows))

    def test_exact_header_and_payload_boundaries(self):
        for frame, overhead in m.FRAMES.items():
            for payload in m.PAYLOAD_BITS:
                for capacity, feasible in [(overhead+payload-1,False),(overhead+payload,True)]:
                    row=next(x for x in m.frame_rows(83,'v1',capacity) if x['framing']==frame and x['payload_bits']==payload)
                    self.assertIs(row['capacity_feasible'],feasible)
                    self.assertEqual(row['delivery_status'],'not_evaluated')

    def test_complete_grid_and_eight_prefix(self):
        selected=m.validate_snapshot(snapshot())
        self.assertEqual(sorted(selected),list(range(83,101)))
        self.assertEqual(sum(v['status']=='completed' for v in selected.values()),8)

    def test_duplicate_identity_rejected(self):
        data=snapshot();data['records'][-1]=copy.deepcopy(data['records'][-2])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            m.validate_snapshot(data)

    def test_added_available_record_requires_new_id(self):
        data=snapshot();next(x for x in data['records'] if x['model_alias']=='tinyllama11' and x['source_order']==91)['status']='completed'
        data['counts']={'completed':9,'not_attempted':291}
        with self.assertRaisesRegex(ValueError,'available prefix'):
            m.validate_snapshot(data)

    def test_bad_denominator_and_false_counts_rejected(self):
        data=snapshot();data['records'].pop()
        with self.assertRaisesRegex(ValueError,'denominator'):
            m.validate_snapshot(data)
        data=snapshot();data['counts']['completed']=9
        with self.assertRaisesRegex(ValueError,'counts'):
            m.validate_snapshot(data)

    def test_input_inventory_rejects_duplicates_and_wrong_commit(self):
        with tempfile.TemporaryDirectory() as path:
            root=pathlib.Path(path)
            with self.assertRaisesRegex(ValueError,'commit'):
                m.bound_inputs(root,dict(commit='wrong',actual_git_readback=True,files=[]))
            with self.assertRaisesRegex(ValueError,'inventory'):
                m.bound_inputs(root,dict(commit=m.COMMIT,actual_git_readback=True,files=[{'path':'same'}]*21))

    def test_unlisted_expected_input_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as path:
            root=pathlib.Path(path);manifest=input_fixture(root)
            self.assertEqual(len(m.bound_inputs(root,manifest)),21)
            manifest['files'][5]['path']='unrelated.json'
            (root/'unrelated.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'expected input inventory'):
                m.bound_inputs(root,manifest)

    def test_symlink_ancestor_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=pathlib.Path(path);manifest=input_fixture(root)
            (root/'data').rename(root/'actual-data')
            (root/'data').symlink_to(root/'actual-data',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'linked input ancestor'):
                m.bound_inputs(root,manifest)


if __name__=='__main__':
    unittest.main(verbosity=2)
