"""Synthetic binding transport checks only; no actual cover/codec evaluation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/package_first80_transport_inputs_v1.py"
spec = importlib.util.spec_from_file_location("binding_package", SOURCE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "root"
        self.root.mkdir()
        P.new(self.root / P.HELPER, SOURCE.read_bytes())
        # These named fixture byte streams intentionally are not real gzip data:
        # packaging verifies exact bytes only, while the scientific runner owns
        # row semantics. They must never be treated as measured input records.
        self.data = {"records.jsonl.gz": b"synthetic-records-only",
                     "input_bound_grid.jsonl.gz": random.Random(80).randbytes(400000)}
        manifest = {"schema": "prospective-first80-input-binding-v1", "experiment_id": P.ID,
            "model_record_pairs": 300, "cells": 15300, "completed_covers": 80,
            "reused_public_conditions": 1440, "new_public_conditions": 480,
            "encode_calls": 0, "decode_calls": 0, "model_calls": 0, "new_outcomes_known": False,
            "files_sha256": {name: P.sha(raw) for name, raw in self.data.items()},
            "synthetic_fixture_only": True}
        self.data["manifest.json"] = P.canonical(manifest) + b"\n"
        for name, raw in self.data.items():
            P.new(self.root / P.BINDING / name, raw)
        self.destination = Path(self.temp.name) / "restored"

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_all_bytes_multichunk(self):
        manifest = P.package(self.root)
        self.assertGreater(manifest["chunk_count"], 1)
        receipt = P.restore(self.root, self.destination)
        self.assertEqual(receipt["files"], 3)
        for name, raw in self.data.items():
            self.assertEqual((self.destination / P.BINDING / name).read_bytes(), raw)

    def test_corrupt_part_does_not_create_destination(self):
        manifest = P.package(self.root)
        part = self.root / manifest["chunks"][0]["path"]
        part.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "chunk identity"):
            P.restore(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_package_and_destination_refused(self):
        P.package(self.root)
        with self.assertRaisesRegex(ValueError, "exists"):
            P.package(self.root)
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "exists"):
            P.restore(self.root, self.destination)

    def test_extra_member_and_changed_binding_refused(self):
        extra = self.root / P.BINDING / "extra"
        extra.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "inventory"):
            P.package(self.root)
        extra.unlink()
        (self.root / P.BINDING / "records.jsonl.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hashes"):
            P.package(self.root)
        self.assertFalse((self.root / P.ART).exists())

    def test_mutated_chunk_order_bool_and_helper_refused(self):
        manifest = P.package(self.root)
        mpath = self.root / P.ART / "manifest.json"
        for mutate in (lambda x: x.update(chunk_count=True),
                       lambda x: x["chunks"][0].update(offset=False),
                       lambda x: x["helper"].update(sha256="0" * 64)):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            mpath.write_bytes(P.canonical(changed))
            with self.assertRaises(ValueError):
                P.restore(self.root, self.destination)
            self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
