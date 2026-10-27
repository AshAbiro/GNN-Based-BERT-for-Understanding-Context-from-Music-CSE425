import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from captions_data import LPMusicCapsDataset, build_task_frame, TEXT_FIELDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lp_musiccaps")


def test_loads_real_parquets():
    ds = LPMusicCapsDataset(data_dir=DATA_DIR)
    assert ds.frames["train"].shape[0] == 18706
    assert ds.frames["validation"].shape[0] == 1825
    assert ds.frames["test"].shape[0] == 5329


def test_caption_attribute_prediction_is_excluded():
    # dataset authors explicitly advise against this field -- must not be a valid text_field choice
    assert "caption_attribute_prediction" not in TEXT_FIELDS


def test_vocab_and_labels_real():
    ds = LPMusicCapsDataset(data_dir=DATA_DIR)
    vocab = ds.vocab("tag_top50")
    assert len(vocab) == 50
    track_ids, text, Y, vocab2, paths = build_task_frame(ds, "train", tag_field="tag_top50")
    assert Y.shape == (len(track_ids), 50)
    assert len(text) == len(track_ids) == len(paths)
    assert (Y.sum(axis=1) > 0).all()      # require_caption filtering guarantees >=1 tag per row
    assert all(len(t) > 0 for t in text)  # and non-empty captions


def test_no_split_leakage_by_track_id():
    ds = LPMusicCapsDataset(data_dir=DATA_DIR)
    train_ids, _, _, vocab, _ = build_task_frame(ds, "train")
    val_ids, _, _, _, _ = build_task_frame(ds, "validation", vocab=vocab)
    test_ids, _, _, _, _ = build_task_frame(ds, "test", vocab=vocab)
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))


if __name__ == "__main__":
    test_loads_real_parquets()
    test_caption_attribute_prediction_is_excluded()
    test_vocab_and_labels_real()
    test_no_split_leakage_by_track_id()
    print("all captions_data tests passed")
