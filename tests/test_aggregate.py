import unittest

from svgfaith.aggregate import unique_ancestor_score
from svgfaith.evaluate import equal_axis_accuracy, outcome


class AggregateTest(unittest.TestCase):
    def test_paper_example(self) -> None:
        # An apple exists with probability 0.8. "The apple is red" depends on it
        # and scores 0.9, so its gated contribution is 0.81.
        values = [0.8, 0.9]
        parents = [[], [0]]
        gated_red = 0.9 * ((1 - 0.5) + 0.5 * 0.8)
        self.assertAlmostEqual(gated_red, 0.81)
        self.assertAlmostEqual(
            unique_ancestor_score(values, parents, alpha=0.5),
            (0.8 + 0.81) / 2,
        )

    def test_ancestor_is_counted_once(self) -> None:
        values = [0.5, 0.5, 1.0]
        parents = [[], [0], [0, 1]]
        score = unique_ancestor_score(values, parents, alpha=1.0)
        self.assertAlmostEqual(score, (0.5 + 0.5 * 0.5 + 1.0 * 0.5 * 0.5) / 3)

    def test_tie_threshold(self) -> None:
        self.assertEqual(outcome(0.50, 0.49, 0.01), "tie")
        self.assertEqual(outcome(0.50, 0.489, 0.01), "win")
        self.assertEqual(outcome(0.40, 0.42, 0.01), "loss")

    def test_equal_axis_mean(self) -> None:
        rows = []
        for axis in (
            "negation",
            "attribute_binding",
            "comparative_size",
            "spatial_relation",
            "object_identity",
            "counting",
            "part_completeness",
            "dense_long_prompt",
        ):
            rows.append({"axis": axis, "score_a": 1.0, "score_b": 0.0})
            rows.append({"axis": axis, "score_a": 0.0, "score_b": 1.0})
        self.assertAlmostEqual(equal_axis_accuracy(rows, 0.01), 0.5)


if __name__ == "__main__":
    unittest.main()
