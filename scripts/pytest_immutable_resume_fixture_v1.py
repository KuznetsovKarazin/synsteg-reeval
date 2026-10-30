"""Build the original-pair fixture for the frozen resume-v3 synthetic tests.

Only pinned immutable inputs are read. The advancing cumulative run is neither
read nor written. This is a pytest fixture adapter, not a production runner.
"""
import hashlib
import importlib.util
import json
from pathlib import Path


HARNESS = Path('results/large-campaign-continuation-v3-audit/test_immutable_baseline.py')
HARNESS_SHA = '08155fa5ecc051d976001ffc5d9ba8b5fccce7406797ef949d1164d93ff403f2'
UTILITY_SHA = 'd3cfc68320e330b89d0f5db2019cf6c33a2460b5c154cbc35f3fc7b6238e7101'
INDEX_SHA = '1cae11c371226b8c753bb4c1959b304882fdbeb9570f8e902024893d9344ec98'


def build_baseline(root, destination):
    """Populate an empty temporary checkout and return its input provenance.

    A changed/missing input raises before any fixture files are written. No
    fallback to the live run is permitted. The reviewed harness exposes pinned
    constants and checked_bytes(), but its subprocess-running main is not used.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Fixture destination must be a separate empty directory')
    harness_path = root / HARNESS
    if harness_path.is_symlink() or not harness_path.is_file():
        raise ValueError('Missing or symlinked immutable-baseline harness')
    harness_bytes = harness_path.read_bytes()
    if hashlib.sha256(harness_bytes).hexdigest() != HARNESS_SHA:
        raise ValueError('Changed pinned immutable-baseline harness')
    spec = importlib.util.spec_from_file_location('_resume_fixture_pinned_harness', harness_path)
    harness = importlib.util.module_from_spec(spec)
    exec(compile(harness_bytes, str(harness_path), 'exec'), harness.__dict__)
    sources = {str(HARNESS): HARNESS_SHA}

    def checked(relative, expected):
        data = harness.checked_bytes(root, relative, expected)
        sources[str(relative)] = expected
        return data

    baseline = json.loads(checked(harness.BASELINE, harness.BASELINE_SHA))
    if baseline['source_main_commit'] != harness.BASELINE_COMMIT or baseline['completed'] != 2:
        raise ValueError('Baseline must be the independently verified original pair')
    manifest_bytes = checked(harness.MANIFEST, harness.MANIFEST_SHA)
    manifest = json.loads(manifest_bytes)
    utility_bytes = checked(harness.UTILITY, UTILITY_SHA)
    if json.loads(utility_bytes)['source_sha256'] != harness.UTILITY_SOURCE_SHA:
        raise ValueError('Changed frozen utility source map')
    inputs = {str(harness.MANIFEST): manifest_bytes, str(harness.UTILITY): utility_bytes}
    for relative in harness.DEPENDENCIES:
        inputs[relative] = checked(relative, manifest['frozen_files_sha256'][relative])
    for relative, expected in harness.UTILITY_SOURCE_SHA.items():
        inputs[relative] = checked(relative, expected)
    marker = (b'"""Independent, offline-first scientific audit utilities."""\n'
              b'\n__version__ = "0.1.0"\n')
    inputs['src/synsteg_audit/__init__.py'] = checked(
        'src/synsteg_audit/__init__.py', hashlib.sha256(marker).hexdigest())
    run_prefix = str(harness.RUN) + '/'
    run_files = {row['path']: row['sha256'] for row in baseline['input_evidence']
                 if row['path'].startswith(run_prefix)}
    for name in ['local_ledger/campaign.lock', 'run_guard/campaign.lock']:
        run_files[str(harness.RUN / name)] = hashlib.sha256(b'0').hexdigest()
    index_path = harness.SNAPSHOT / 'files.json'
    if json.loads(checked(index_path, INDEX_SHA)) != run_files:
        raise ValueError('Immutable snapshot index differs from original-pair evidence')
    actual_paths = {str(p.relative_to(root / harness.SNAPSHOT))
                    for p in (root / harness.SNAPSHOT / harness.RUN).rglob('*') if p.is_file()}
    if actual_paths != set(run_files):
        raise ValueError('Missing or additional immutable snapshot files')
    for relative, expected in run_files.items():
        inputs[relative] = checked(harness.SNAPSHOT / relative, expected)
    # All evidence is validated before creating any checkout content.
    for relative, data in inputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {'schema': 'pytest-immutable-resume-fixture-v1',
            'fixture': str(harness.SNAPSHOT), 'baseline_completed': 2,
            'baseline_commit': harness.BASELINE_COMMIT,
            'source_files_sha256': sources,
            'checkout_files_sha256': {p: hashlib.sha256(b).hexdigest() for p, b in inputs.items()},
            'live_cumulative_run_used': False}
