#!/usr/bin/env python3
"""Generate 5-fold ESC-50 manifests for SSLAM/Fairseq."""

import argparse
import csv
import wave
from pathlib import Path


def get_audio_frames(path):
    """Return the number of audio frames in a WAV file."""
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes()


def write_split(rows, audio_dir, output_dir, split):
    """Write Fairseq TSV and label files."""
    tsv_path = output_dir / f"{split}.tsv"
    lbl_path = output_dir / f"{split}.lbl"

    with tsv_path.open("w", encoding="utf-8") as tsv, \
         lbl_path.open("w", encoding="utf-8") as lbl:

        # First line: absolute audio directory
        tsv.write(str(audio_dir.resolve()) + "\n")

        for row in rows:
            filename = row["filename"]
            label = int(row["target"])

            audio_path = audio_dir / filename

            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)

            frames = get_audio_frames(audio_path)

            tsv.write(f"{filename}\t{frames}\n")

            # The SSLAM loader reads labels from field 2.
            lbl.write(f"{filename}\t{label}\n")


def prepare_esc50(dataset_root, output_root):
    """Prepare five ESC-50 cross-validation folds."""
    audio_dir = dataset_root / "audio"
    metadata_path = dataset_root / "meta" / "esc50.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2000, "Expected 2000 ESC-50 samples"

    classes = sorted({int(row["target"]) for row in rows})
    assert classes == list(range(50)), "Expected labels 0-49"

    for fold in range(1, 6):
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_rows = [
            row for row in rows
            if int(row["fold"]) != fold
        ]

        eval_rows = [
            row for row in rows
            if int(row["fold"]) == fold
        ]

        assert len(train_rows) == 1600
        assert len(eval_rows) == 400

        write_split(
            train_rows, audio_dir, fold_dir, "train"
        )

        write_split(
            eval_rows, audio_dir, fold_dir, "eval"
        )

        # SSLAM label descriptor format: index,label
        descriptor_path = fold_dir / "label_descriptors.csv"

        with descriptor_path.open(
            "w", encoding="utf-8"
        ) as f:
            for class_id in classes:
                f.write(f"{class_id},{class_id}\n")

        print(
            f"Fold {fold}: "
            f"Train={len(train_rows)}, "
            f"Eval={len(eval_rows)}"
        )

    print("ESC-50 manifest preparation complete.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of ESC-50"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output manifest directory"
    )

    args = parser.parse_args()

    prepare_esc50(
        dataset_root=args.dataset_root.resolve(),
        output_root=args.output_dir.resolve()
    )


if __name__ == "__main__":
    main()