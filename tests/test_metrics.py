import unittest

import numpy as np

from musiccaps_ml.metrics import bidirectional_retrieval, multilabel_f1


class MetricsTests(unittest.TestCase):
    def test_perfect_multilabel_f1(self):
        target = np.array([[1, 0], [0, 1]])
        probability = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.assertEqual(multilabel_f1(target, probability), {"macro_f1": 1.0, "micro_f1": 1.0})

    def test_perfect_bidirectional_retrieval(self):
        embedding = np.eye(12)
        metrics = bidirectional_retrieval(embedding, embedding)
        for direction in metrics.values():
            self.assertEqual(direction, {"R@1": 1.0, "R@5": 1.0, "R@10": 1.0})

    def test_retrieval_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            bidirectional_retrieval(np.eye(2), np.eye(3))


if __name__ == "__main__":
    unittest.main()

