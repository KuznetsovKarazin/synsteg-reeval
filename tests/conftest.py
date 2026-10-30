"""Use an immutable original pair for the frozen resume-v3 test module only.

All original test functions and node IDs remain collected. Other test modules
retain their own ROOT/REPO and fixtures. See docs/pytest_immutable_resume_fixture_v1_ru.md.
"""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEST = ROOT / 'tests/test_large_campaign_resume_v3.py'


@pytest.fixture(scope='module', autouse=True)
def _immutable_original_pair_for_frozen_resume_tests(request, tmp_path_factory):
    module = request.module
    if Path(module.__file__).resolve() != FROZEN_TEST:
        yield
        return
    helper_path = ROOT / 'scripts/pytest_immutable_resume_fixture_v1.py'
    spec = importlib.util.spec_from_file_location('_resume_pytest_fixture_v1', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    baseline_root = tmp_path_factory.mktemp('resume-v3-original-pair')
    provenance = helper.build_baseline(ROOT, baseline_root)
    previous_root = module.REPO
    if previous_root != ROOT:
        raise ValueError('Frozen resume test REPO was already redirected')
    nodeids = [item.nodeid for item in request.session.items
               if Path(item.path).resolve() == FROZEN_TEST]
    request.config._synsteg_resume_fixture_v1 = {**provenance, 'selected_nodeids': nodeids}
    # Explicit fixture-root selection, scoped to this one frozen test module.
    # No frozen source, live run path, function body or OS file operation is mocked.
    module.REPO = baseline_root
    try:
        yield
    finally:
        module.REPO = previous_root


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    provenance = getattr(config, '_synsteg_resume_fixture_v1', None)
    if provenance:
        terminalreporter.write_line(
            'resume-v3 fixture: {} original test(s), pinned immutable original pair; '
            'live cumulative run not used'.format(len(provenance['selected_nodeids'])))
