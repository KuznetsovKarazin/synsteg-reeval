from pathlib import Path
import importlib.util
import json
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prospective_stages", ROOT / "scripts/run_prospective_stages.py")
stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stages)


class ProspectiveStageCheckpoints(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.run = self.base / "pilot"
        self.run.mkdir()
        self.control = self.base / "control"
        self.control.mkdir()
        (self.run / "freeze.json").write_text('{"freeze_id":"abc"}\n')

    def checkpoint(self):
        return stages.checkpoint(self.run, self.control, self.base / "checkpoints", "frozen_design", "abc", "1" * 40)

    def test_checkpoint_preserves_attempt_markers_and_rejects_tampering(self):
        marker = self.run / "attempt_started/train/first.json"
        marker.parent.mkdir(parents=True)
        marker.write_text('{"started":true}\n')
        _, manifest = self.checkpoint()
        archive_path = Path(manifest["archive_path"])
        with tarfile.open(archive_path, "r:gz") as archive:
            self.assertEqual(archive.extractfile("pilot/attempt_started/train/first.json").read(), marker.read_bytes())
        self.assertEqual(manifest["files_sha256"]["attempt_started/train/first.json"], stages.sha256(marker))
        with archive_path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "archive hash"):
            self.checkpoint()

    def test_remote_receipt_requires_exact_verified_bytes(self):
        _, manifest = self.checkpoint()
        receipt = {"archive_sha256": manifest["archive_sha256"], "verified_remote_sha256": manifest["archive_sha256"],
                   "freeze_id": "abc", "stage": "frozen_design", "remote_uri": "git:repo/commit/path",
                   "verification_evidence": "Retrieved remote archive and checked SHA256", "verified_at_unix": 1.0}
        stages.validate_receipt(receipt, manifest)
        receipt["verified_remote_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "verified checkpoint bytes"):
            stages.validate_receipt(receipt, manifest)

    def test_reuse_allows_ledger_append_but_rejects_changed_records(self):
        (self.run / "events.jsonl").write_text('{"event":"frozen"}\n')
        (self.run / "record.json").write_text('{"original":true}\n')
        self.checkpoint()
        with (self.run / "events.jsonl").open("a") as stream:
            stream.write('{"event":"later_stage"}\n')
        self.checkpoint()
        (self.run / "record.json").write_text('{"original":false}\n')
        with self.assertRaisesRegex(RuntimeError, "Previously checkpointed file changed"):
            self.checkpoint()

    def test_writer_lock_recovery_requires_known_exited_owner(self):
        lock = self.run / ".writer.lock"
        lock.write_text("124")
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            stages.recover_known_exited_child_lock(self.run, 123, "train_generation")
        self.assertEqual(lock.read_text(), "124")
        stages.recover_known_exited_child_lock(self.run, 124, "train_generation")
        self.assertFalse(lock.exists())
        self.assertEqual(json.loads((self.run / "events.jsonl").read_text())["stale_pid"], 124)

    def test_invalid_evaluation_is_preserved_and_not_executed_again(self):
        (self.run / "evaluation.json").write_text("{}")
        with patch.object(stages.subprocess, "Popen", side_effect=AssertionError("must not rerun evaluation")):
            with self.assertRaisesRegex(RuntimeError, "invalid identity"):
                stages.execute_stage(ROOT, self.run, self.control, "test_evaluation", ("evaluate",))
        self.assertEqual((self.run / "evaluation.json").read_text(), "{}")

    def test_generation_completion_check_does_not_open_record_text(self):
        (self.run / "manifests").mkdir()
        (self.run / "manifests/test.json").write_text('[{"id":"one"}]')
        (self.run / "records/test").mkdir(parents=True)
        (self.run / "records/test/one.json").write_text("This is deliberately not parsed as JSON here.")
        (self.run / "test_generation.json").write_text(json.dumps({"complete": True, "records": 1, "expected": 1,
                                                                  "records_sha256": "opaque"}))
        self.assertTrue(stages.stage_complete(self.run, self.control, "test_generation"))

    def test_missing_remote_receipt_stops_at_timeout(self):
        manifest_path, manifest = self.checkpoint()
        with self.assertRaises(TimeoutError):
            stages.await_receipt(self.control, manifest_path, manifest, 0.01)

    def test_sigterm_waits_for_owned_child_and_preserves_attempt_marker(self):
        fake_repo = self.base / "fake_repo"
        (fake_repo / "scripts").mkdir(parents=True)
        (fake_repo / "scripts/prospective_experiment.py").write_text(
            "import os,sys,time\nfrom pathlib import Path\n"
            "run=Path(sys.argv[sys.argv.index('--run')+1])\n"
            "(run/'.writer.lock').write_text(str(os.getpid()))\n"
            "(run/'attempt_started').mkdir(exist_ok=True)\n"
            "(run/'attempt_started/kept.json').write_text('{}')\n"
            "while True: time.sleep(1)\n")
        checkpoints = self.base / "signal_checkpoints"
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/run_prospective_stages.py"),
            "--repo", str(fake_repo), "--run", str(self.run), "--freeze-id", "abc",
            "--source-commit", "1" * 40, "--checkpoint-directory", str(checkpoints),
            "--continuity-evidence", "isolated no-model signal test"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            marker = self.run / "attempt_started/kept.json"
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=15), 143)
            child = json.loads((checkpoints / "pilot-controller/owned_child.json").read_text())
            self.assertEqual(child["status"], "exited_after_controller_signal")
            self.assertEqual(child["returncode"], -signal.SIGTERM)
            self.assertTrue(marker.exists())
            self.assertFalse((self.run / ".writer.lock").exists())
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            process.stderr.close()


class ProspectiveV2Design(unittest.TestCase):
    def test_new_design_preserves_method_and_fixes_lengths(self):
        old = json.loads((ROOT / "configs/prospective_pilot_cpu_v1.json").read_text())
        new = json.loads((ROOT / "configs/prospective_pilot_cpu_v2.json").read_text())
        for field in ("method_config", "attack", "generation", "model", "payload_bits", "repetitions", "maximum_attempts"):
            self.assertEqual(new[field], old[field], field)
        self.assertEqual((new["generation_seed"], new["payload_seed"], new["statistics"]["bootstrap_seed"]),
                         (2026102301, 2026102302, 2026102303))
        self.assertEqual([len(new["topics"][split]) * 3 * 2 for split in ("train", "development", "test")], [24, 12, 24])
        old_topics = {item["text"] for items in old["topics"].values() for item in items}
        new_topics = [item["text"] for items in new["topics"].values() for item in items]
        self.assertEqual(len(set(new_topics)), 10)
        self.assertFalse(set(new_topics) & old_topics)
        old_ids = {prompt["family_id"] for split in old["prompts"].values() for prompt in split.values()}
        prompts = [prompt for split in new["prompts"].values() for prompt in split.values()]
        self.assertEqual(len({prompt["family_id"] for prompt in prompts}), 9)
        self.assertFalse(old_ids & {prompt["family_id"] for prompt in prompts})
        for prompt in prompts:
            self.assertIn("180", prompt["template"])
            self.assertNotIn("280", prompt["template"])


if __name__ == "__main__":
    unittest.main()
