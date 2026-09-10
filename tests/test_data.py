import unittest

import numpy as np

from musiccaps_ml.data import encode_tags, make_split, parse_tags


class DataTests(unittest.TestCase):
    def test_tag_parsing(self):
        self.assertEqual(parse_tags("/m/a,/m/b", "audioset"), ["/m/a", "/m/b"])
        self.assertEqual(parse_tags("['Piano', 'slow tempo']", "aspects"), ["piano", "slow tempo"])

    def test_multihot_encoding(self):
        encoded = encode_tags([["a", "b"], ["b"]], ["a", "b"])
        np.testing.assert_array_equal(encoded, [[1, 1], [0, 1]])

    def test_split_is_disjoint_and_complete(self):
        labels = np.tile(np.eye(3, dtype=np.float32), (20, 1))
        split, _ = make_split(labels, seed=7)
        self.assertEqual(len(split), len(labels))
        self.assertEqual(set(split), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()

