#!/usr/bin/env python3
"""One reproducible synthetic preservation roundtrip; no actual trace reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/package_candidate_mask_order_v1.py"
spec = importlib.util.spec_from_file_location("mask_order_packager", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="mask-order-synthetic-package-") as temp:
    root = Path(temp)
    for path in module.HELPERS:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / path).read_bytes())
    result_dir = root / module.RESULT
    result_dir.mkdir(parents=True)
    # Intentionally not scientific rows. Counts in the synthetic summary are
    # only fixture values needed to exercise the packager's closed-run gate.
    payload = random.Random(20261026).randbytes(600000).hex()
    rows = (json.dumps({"synthetic_packaging_fixture_only": True, "payload": payload}) + "\n").encode()
    (result_dir / "probe-comparisons.jsonl").write_bytes(rows)
    summary = {"experiment_id": module.ID, "factual_probes": 361, "recorded_rows": 18050,
               "scope": {"native_calls": 0}, "probe_rows_sha256": hashlib.sha256(rows).hexdigest(),
               "synthetic_packaging_fixture_only": True,
               "denominator_fields_are_fixture_values_not_measurements": True}
    (result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    (result_dir / "report_ru.md").write_text("Synthetic packaging fixture only; no scientific results.\n")
    (result_dir / "optional-binary.dat").write_bytes(b"\xffsynthetic binary fixture")
    status = module.package(root)
    assert status["chunk_count"] > 1
    assert status["primary_file_count"] == 4
    plan = json.loads((root / module.OUT / "save-plan.json").read_text())
    assert plan["archive_only_primary_paths"] == [(module.RESULT / name).as_posix()
                                                   for name in ("optional-binary.dat", "probe-comparisons.jsonl")]
    assert plan["include_this_plan"] is True
    before = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file()}
    try:
        module.package(root)
    except ValueError as exc:
        assert str(exc) == "existing package refused"
    else:
        raise AssertionError("second preservation overwritten")
    after = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    assert before == after
    print(json.dumps({"schema": "author-candidate-mask-order-v1-synthetic-package-check",
        "status": "passed", "fixture_only": True, "actual_trace_analysis_runs": 0,
        "native_calls": 0, "fixture_seed": 20261026, "fixture_payload_random_bytes": 600000,
        "fixture_primary_files": 4, "fixture_chunk_count": status["chunk_count"],
        "all_four_primary_files_restored_exact": True, "non_utf8_archived_only": True,
        "own_save_plan_inclusion_explicit": True, "existing_output_refused_without_change": True,
        "helper_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "frozen_generic_helpers": module.HELPERS}, indent=2))
