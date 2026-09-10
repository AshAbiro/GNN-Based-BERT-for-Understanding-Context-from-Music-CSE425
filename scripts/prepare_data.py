from __future__ import annotations

import argparse
from pathlib import Path

from musiccaps_ml.data import prepare_experiment_data


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare shared MusicCaps labels and splits")
    parser.add_argument("--csv", default="musiccaps_with_audio_paths.csv")
    parser.add_argument("--audio-dir", default="audio_clips")
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--target-source", choices=("audioset", "aspects"), default="audioset")
    parser.add_argument("--min-frequency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = prepare_experiment_data(
        csv_path=args.csv,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        target_source=args.target_source,
        min_frequency=args.min_frequency,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        force=args.force,
    )
    counts = data.frame["split"].value_counts().to_dict()
    print(f"Prepared {len(data.frame)} rows and {len(data.vocabulary)} labels")
    print(f"Split method: {data.split_method}; counts: {counts}")
    print(f"Artifacts: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

