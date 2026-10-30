#!/usr/bin/env python3
"""Run the frozen quality helper with a transparent path-type compatibility fix.

The first export failed before writing a corpus: the helper passes __file__ as
a string to the inherited Path-only file_hash function. Keep all frozen files
unchanged; normalize only this function's argument and identify this launcher
in the secondary export manifest. Models, cohorts and arithmetic are unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
HELPER = HERE.with_name('crossmodel_quality.py')
spec = importlib.util.spec_from_file_location('crossmodel_quality_frozen_pathfix', HELPER)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
original_hash = helper.file_hash
helper.file_hash = lambda path: original_hash(Path(path))
original_export = helper.make_export


def make_export(run):
    manifest, rows, corpus, records = original_export(run)
    manifest['postfreeze_compatibility_launcher'] = {
        'file': str(HERE.relative_to(helper.ROOT)),
        'sha256': hashlib.sha256(HERE.read_bytes()).hexdigest(),
        'frozen_helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        'change': 'Normalize file_hash arguments to pathlib.Path; no changes to data, models, cohorts, or arithmetic.',
        'reason': 'Initial export failed with AttributeError on string __file__ before corpus creation or inference.'}
    return manifest, rows, corpus, records


helper.make_export = make_export

if __name__ == '__main__':
    helper.main()
