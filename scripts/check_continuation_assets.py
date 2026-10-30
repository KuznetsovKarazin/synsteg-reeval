#!/usr/bin/env python3
"""Inventory the saved large-panel design without network access or generation.

This is an inventory, not the large-panel runner or an execution authorization.
Only named project assets and matching delivery ZIPs are inspected. Credentials,
environment values, model outputs and test texts are not read into the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BACKUP_NAME = 'synsteg-shared-dictionary-pilot-v1.zip'
BACKUP_SHA256 = '4d11d395dcc334da1761daa49bf72f97ca541ef17e43cd647d453f92e87a5742'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    checks = []
    paths = ['configs/shared_dictionary_large_panel_v1.design.json',
             'src/synsteg_audit/shared_dictionary.py',
             'data/shared_dictionary_v1.json',
             'data/shared_dictionary_ai_v2.json',
             'scripts/build_shared_dictionary.py',
             'scripts/replay_shared_dictionary_ai_v2.py',
             'scripts/review_shared_dictionary_ai.py']
    for name in paths:
        path = root / name
        checks.append({'path': name, 'present': path.is_file(),
                       'bytes': path.stat().st_size if path.is_file() else None,
                       'sha256': file_hash(path) if path.is_file() else None})
    design_path = root / paths[0]
    design = json.loads(design_path.read_text(encoding='utf-8')) if design_path.is_file() else {}
    groups = design.get('source_groups', {})
    manifest_name = groups.get('manifest')
    manifest = None
    if isinstance(manifest_name, str) and manifest_name:
        candidate = (root / manifest_name).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            manifest = candidate
    expected = groups.get('manifest_sha256')
    actual = file_hash(manifest) if manifest is not None else None
    backups = []
    candidates = [root / BACKUP_NAME]
    deliveries = root / 'deliveries'
    if deliveries.is_dir():
        candidates.extend(deliveries.rglob(BACKUP_NAME))
    for path in sorted(set(candidates)):
        if path.is_file() and path.resolve().is_relative_to(root):
            digest = file_hash(path)
            backups.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                            'sha256': digest, 'matches_recorded_prior_delivery': digest == BACKUP_SHA256})
    return {
        'schema': 'synsteg-continuation-assets-inventory-v1',
        'network_requests': 0, 'generation_attempts': 0,
        'scope': 'Named local assets only; not a validation of experimental readiness or a search outside this repository.',
        'files': checks,
        'missing_named_assets': [row['path'] for row in checks if not row['present']],
        'source_manifest_declared': bool(manifest_name),
        'source_manifest_present': manifest is not None,
        'source_manifest_hash_matches': bool(expected and actual and expected == actual),
        'design_execution_enabled': design.get('execution_enabled') is True,
        'budget_cap_in_saved_design_usd': design.get('claude_extension', {}).get('budget_cap_usd'),
        'prior_delivery_candidates': backups,
        'execution_authorized_by_this_inventory': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, help='Create a new JSON file; never overwrite one.')
    args = parser.parse_args()
    report = inventory(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
