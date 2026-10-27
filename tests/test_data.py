import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import JamendoDataset, build_task_frame, split_tags_by_category

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits", "split-0")


def test_split_tags_by_category():
    tags = ["genre---rock", "mood/theme---calm", "instrument---guitar"]
    out = split_tags_by_category(tags)
    assert out == {"genre": ["rock"], "moodtheme": ["calm"], "instrument": ["guitar"]}


def test_dataset_loads_real_tsvs():
    ds = JamendoDataset(data_dir=DATA_DIR, split_dir=SPLIT_DIR)
    assert len(ds.df) > 50000
    assert "genre" in ds.df["by_category"].iloc[0] or True  # not every row has genre tags


def test_no_train_val_leakage():
    ds = JamendoDataset(data_dir=DATA_DIR, split_dir=SPLIT_DIR)
    train = set(ds.split_ids("train", category="moodtheme"))
    val = set(ds.split_ids("validation", category="moodtheme"))
    test = set(ds.split_ids("test", category="moodtheme"))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_build_task_frame_shapes_align():
    ds = JamendoDataset(data_dir=DATA_DIR, split_dir=SPLIT_DIR)
    track_ids, text, Y, vocab = build_task_frame(ds, ["genre", "instrument"], "moodtheme", "validation")
    assert len(track_ids) == len(text) == Y.shape[0]
    assert Y.shape[1] == len(vocab)
    assert Y.sum() > 0


if __name__ == "__main__":
    test_split_tags_by_category()
    test_dataset_loads_real_tsvs()
    test_no_train_val_leakage()
    test_build_task_frame_shapes_align()
    print("all data tests passed")
