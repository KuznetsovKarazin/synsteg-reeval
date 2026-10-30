"""Reference categories distinguish uncertainty from a safety certificate."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import assess_nli_diagnostic_v1 as assessment


def flags(meaning=False, grammar=False, ambiguous=False):
    return dict(zip(assessment.FLAGS, (meaning, grammar, ambiguous)))


@pytest.mark.parametrize('left,right,expected', [
    (flags(), flags(), 'no_flag_by_either_not_certified'),
    (flags(ambiguous=True), flags(), 'ambiguity_or_register_only'),
    (flags(), flags(ambiguous=True), 'ambiguity_or_register_only'),
    (flags(meaning=True), flags(), 'one_assistant_definite'),
    (flags(), flags(grammar=True), 'one_assistant_definite'),
    (flags(meaning=True), flags(ambiguous=True), 'one_assistant_definite'),
    (flags(meaning=True), flags(grammar=True), 'both_assistants_definite'),
    (flags(grammar=True, ambiguous=True), flags(meaning=True), 'both_assistants_definite'),
])
def test_reference_category_retains_uncertainty_and_disagreement(left, right, expected):
    assert assessment.category(left, right) == expected
