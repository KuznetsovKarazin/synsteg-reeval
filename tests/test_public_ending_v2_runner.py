"""Source/gate/fixture tests only; no native process or tokenizer is invoked."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ending_runner", ROOT / "scripts/run_public_ending_v2.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunnerSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def test_static_closure_includes_relative_lazy_transitive_imports(self):
        self.write("entry.py", "from synsteg_audit import a\n")
        self.write("src/synsteg_audit/__init__.py", "")
        self.write("src/synsteg_audit/a.py", "def lazy():\n from .b import value\n")
        self.write("src/synsteg_audit/b.py", "from . import c\nvalue=1\n")
        self.write("src/synsteg_audit/c.py", "import math\n")
        self.assertEqual(RUNNER.source_closure(self.root, ["entry.py"]), [
            "entry.py", "src/synsteg_audit/__init__.py", "src/synsteg_audit/a.py",
            "src/synsteg_audit/b.py", "src/synsteg_audit/c.py"])

    def test_missing_local_dependency_refuses(self):
        self.write("entry.py", "from synsteg_audit.missing import value\n")
        with self.assertRaisesRegex(ValueError, "missing local dependency"):
            RUNNER.source_closure(self.root, ["entry.py"])

    def test_unsafe_closure_root_refuses(self):
        with self.assertRaisesRegex(ValueError, "unsafe source"):
            RUNNER.source_closure(self.root, ["../outside.py"])

    def test_proposal_remains_disabled_with_full_denominators(self):
        proposal = RUNNER.prepared(ROOT)
        self.assertIs(proposal["execution_enabled"], False)
        self.assertEqual((proposal["generations"], proposal["fresh_receivers"], proposal["quality_denominator"]), (8, 4, 8))
        self.assertEqual(proposal["source_orders"], [13, 16, 19, 22])
        self.assertFalse(proposal["closed_test_allowed"])
        self.assertFalse(proposal["paid_calls_allowed"])

    def test_selected_record_mutation_refuses(self):
        src = ROOT / RUNNER.EXP
        dest = self.root / RUNNER.EXP
        shutil.copytree(src, dest)
        path = dest / "selected-context-texts.json"
        path.write_bytes(path.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError, "selected development"):
            RUNNER.prepared(self.root)

    def test_proposal_activation_cannot_edit_disabled_input(self):
        src = ROOT / RUNNER.EXP
        dest = self.root / RUNNER.EXP
        shutil.copytree(src, dest)
        path = dest / "protocol.proposed.json"
        path.write_bytes(path.read_bytes().replace(b'"execution_enabled":false', b'"execution_enabled":true'))
        with self.assertRaisesRegex(ValueError, "disabled proposal"):
            RUNNER.prepared(self.root)

    def test_all_fixed_case_slots_and_unique_nonces(self):
        cases = RUNNER.fixtures()
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({(c["key_hex"], c["nonce_hex"]) for c in cases}), 8)
        self.assertEqual([c["arm"] for c in cases], ["stego", "ordinary"]*4)
        self.assertEqual({c["payload_hex"] for c in cases if c["arm"] == "stego"}, {bytes(range(16)).hex()})


if __name__ == "__main__":
    unittest.main()
