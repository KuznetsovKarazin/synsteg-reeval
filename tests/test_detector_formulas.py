import itertools
import math
import unittest

import numpy as np

from synsteg_audit.detector_formulas import binoculars_score, fast_detectgpt_analytic


class DetectorFormulaTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.log([[[.8, .2], [.4, .6], [.5, .5]]])
        self.perf = np.log([[[.7, .3], [.1, .9], [.9, .1]]])
        self.ids = np.array([[0, 1, 0]])
        self.mask = np.ones((1, 3), dtype=int)

    def test_binoculars_direction_and_final_distribution(self):
        # Explicit arithmetic independent of the implementation's vectorization.
        nll = -(math.log(.3) + math.log(.1)) / 2
        ce = -( .8 * math.log(.7) + .2 * math.log(.3)
              + .4 * math.log(.1) + .6 * math.log(.9)
              + .5 * math.log(.9) + .5 * math.log(.1)) / 3
        r = binoculars_score(self.obs, self.perf, self.ids, self.mask)
        self.assertAlmostEqual(r.score[0], nll / ce, places=13)
        self.assertEqual(r.n_nll_positions[0], 2)
        self.assertEqual(r.n_cross_entropy_positions[0], 3)
        self.assertEqual(r.ai_score[0], -r.score[0])
        reversed_score = binoculars_score(self.perf, self.obs, self.ids, self.mask).score[0]
        self.assertNotAlmostEqual(r.score[0], reversed_score, places=8)
        changed = self.perf.copy()
        changed[0, -1] = np.log([.5, .5])
        r2 = binoculars_score(self.obs, changed, self.ids, self.mask)
        self.assertEqual(r.performer_nll[0], r2.performer_nll[0])
        self.assertNotEqual(r.score[0], r2.score[0])

    def test_same_positions_ablation_is_explicit_and_final_token_invariant(self):
        r = binoculars_score(self.obs, self.perf, self.ids, self.mask,
                             cross_entropy_positions="next_token")
        changed = self.perf.copy()
        changed[0, -1] = np.log([.5, .5])
        r2 = binoculars_score(self.obs, changed, self.ids, self.mask,
                              cross_entropy_positions="next_token")
        np.testing.assert_equal(r.score, r2.score)
        self.assertEqual(r.n_cross_entropy_positions[0], 2)

    def test_fastdetectgpt_matches_exhaustive_sampling_distribution(self):
        # Enumerate all independent two-token samples rather than reproducing
        # the analytic variance implementation in the test.
        ref = [[.8, .2], [.4, .6]]
        scoring = [[.7, .3], [.1, .9]]
        samples = []
        for labels in itertools.product(range(2), repeat=2):
            prob = ref[0][labels[0]] * ref[1][labels[1]]
            ll = math.log(scoring[0][labels[0]]) + math.log(scoring[1][labels[1]])
            samples.append((prob, ll))
        mean = sum(p * v for p, v in samples)
        var = sum(p * (v - mean) ** 2 for p, v in samples)
        observed = math.log(.3) + math.log(.1)
        expected = (observed - mean) / math.sqrt(var)
        actual = fast_detectgpt_analytic(self.obs, self.perf, self.ids, self.mask)[0]
        self.assertAlmostEqual(actual, expected, places=13)

    def test_fastdetectgpt_scales_as_sqrt_number_of_positions(self):
        one = np.log([[[.8, .2], [.8, .2]]])
        nine = np.repeat(one[:, :1], 10, axis=1)
        a = fast_detectgpt_analytic(one, one, np.zeros((1, 2), int), np.ones((1, 2)))[0]
        b = fast_detectgpt_analytic(nine, nine, np.zeros((1, 10), int), np.ones((1, 10)))[0]
        self.assertAlmostEqual(b, 3 * a, places=13)

    def test_right_padding_does_not_change_either_score(self):
        # Nonfinite pad logits and invalid pad IDs are harmless when masked.
        obs = np.concatenate([self.obs, np.full((1, 2, 2), np.nan)], axis=1)
        perf = np.concatenate([self.perf, np.full((1, 2, 2), np.nan)], axis=1)
        ids = np.concatenate([self.ids, [[-100, -100]]], axis=1)
        mask = np.array([[1, 1, 1, 0, 0]])
        for fn in [fast_detectgpt_analytic, lambda *a: binoculars_score(*a).score]:
            np.testing.assert_allclose(fn(obs, perf, ids, mask),
                                       fn(self.obs, self.perf, self.ids, self.mask), atol=1e-14)

    def test_left_padding_excludes_prediction_from_pad_to_first_real_token(self):
        obs = np.concatenate([np.full((1, 2, 2), np.nan), self.obs], axis=1)
        perf = np.concatenate([np.full((1, 2, 2), np.nan), self.perf], axis=1)
        ids = np.concatenate([[[-100, -100]], self.ids], axis=1)
        mask = np.array([[0, 0, 1, 1, 1]])
        for fn in [fast_detectgpt_analytic, lambda *a: binoculars_score(*a).score]:
            np.testing.assert_allclose(fn(obs, perf, ids, mask),
                                       fn(self.obs, self.perf, self.ids, self.mask), atol=1e-14)

    def test_batch_matches_individual_and_genuine_eos_retained(self):
        r = binoculars_score(np.repeat(self.obs, 2, axis=0), np.repeat(self.perf, 2, axis=0),
                              np.repeat(self.ids, 2, axis=0), np.repeat(self.mask, 2, axis=0))
        self.assertEqual(r.score[0], r.score[1])
        # Token 0 may be a real EOS; default attention masking retains it.
        ordinary = binoculars_score(self.obs, self.perf, self.ids, self.mask)
        upstream_id_mask = binoculars_score(self.obs, self.perf, self.ids, self.mask, pad_token_id=0)
        self.assertEqual(ordinary.n_cross_entropy_positions[0], 3)
        self.assertEqual(upstream_id_mask.n_cross_entropy_positions[0], 1)

    def test_input_failures_are_explicit(self):
        for fn in [binoculars_score, fast_detectgpt_analytic]:
            with self.subTest(fn=fn), self.assertRaises(ValueError):
                fn(self.obs, self.perf, self.ids, [[0, 0, 0]])
            with self.subTest(fn=fn), self.assertRaises(ValueError):
                fn(self.obs, self.perf, self.ids, [[1, .5, 1]])
            with self.subTest(fn=fn), self.assertRaises(ValueError):
                fn(self.obs, self.perf, [[0, 2, 0]], self.mask)
        with self.assertRaises(ValueError):
            fast_detectgpt_analytic(np.zeros((1, 3, 2)), np.zeros((1, 3, 2)), self.ids, self.mask)
        with self.assertRaises(ValueError):
            binoculars_score(self.obs, np.zeros((1, 3, 3)), self.ids, self.mask)


if __name__ == "__main__":
    unittest.main()
