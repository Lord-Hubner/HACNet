"""
Export HACNet-generated images for every fold-assigned training row in the TCGA datasets.

The script expects each dataset directory to contain:

    data_<suffix>
    label_<suffix>
    fold_assignments.csv

Example:

    python src/export_images.py --checkpoint path/to/actor.pth
    python src/export_images.py --dataset TCGA-KIRC --checkpoint path/to/model.pth

If no checkpoint is supplied, HACNet is initialized with random weights. That is
mainly useful for smoke tests; thesis exports should use a trained checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from actors.propose import HACNet


TCGA_PREFIX = "TCGA-"
DEFAULT_OUTPUT_DIR = Path("datasets") / "HACNet_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HACNet images and save them by StratifiedKFold fold."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Directory containing the TCGA-* dataset folders.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset folder name to export, e.g. TCGA-KIRC. "
            "Can be passed multiple times. Defaults to all TCGA-* folders."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Path to actor.pth, model.pth, or best_checkpoint.pth. "
            "Without this, the actor is randomly initialized."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory where exported images and metadata are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of rows to convert per forward pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for generation.",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=100,
        help="Image side length used when no checkpoint is supplied.",
    )
    parser.add_argument(
        "--reg-coef",
        type=float,
        default=10.0,
        help="HACNet regularization coefficient used when building the actor.",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=10.0,
        help="HACNet initial temperature used when building the actor.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=0.66,
        help="HACNet final temperature used when building the actor.",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=8000,
        help="HACNet max_iteration used when building the actor.",
    )
    parser.add_argument(
        "--scaler",
        choices=("minmax", "standard", "none"),
        default="minmax",
        help="Feature scaling applied before image generation.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Save images directly under fold directories instead of class subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if the output PNG already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Export only the first N fold-assigned samples per dataset. Useful for smoke tests.",
    )
    return parser.parse_args()


def discover_datasets(datasets_root: Path, requested: Optional[List[str]]) -> List[Path]:
    if requested:
        dataset_dirs = [datasets_root / name for name in requested]
    else:
        dataset_dirs = sorted(
            path for path in datasets_root.iterdir() if path.is_dir() and path.name.startswith(TCGA_PREFIX)
        )

    missing = [path for path in dataset_dirs if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset folder(s) not found: {names}")

    return dataset_dirs


def dataset_suffix(dataset_dir: Path) -> str:
    return dataset_dir.name.replace(TCGA_PREFIX, "")


def dataset_files(dataset_dir: Path) -> Tuple[Path, Path, Path]:
    suffix = dataset_suffix(dataset_dir)
    data_path = dataset_dir / f"data.csv"
    label_path = dataset_dir / f"label.csv"
    folds_path = dataset_dir / "fold_assignments.csv"

    for path in (data_path, label_path, folds_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    return data_path, label_path, folds_path


def load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data_path, label_path, folds_path = dataset_files(dataset_dir)

    data = pd.read_csv(data_path, index_col=0)
    labels_df = pd.read_csv(label_path, index_col=0)
    folds = pd.read_csv(folds_path)

    if labels_df.shape[1] != 1:
        raise ValueError(f"Expected one label column in {label_path}, found {labels_df.shape[1]}")

    labels = labels_df.iloc[:, 0]
    if len(data) != len(labels):
        raise ValueError(
            f"{dataset_dir.name}: data has {len(data)} rows, labels have {len(labels)} rows"
        )

    required_fold_columns = {"sample_idx", "fold"}
    missing_columns = required_fold_columns.difference(folds.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{folds_path} is missing required column(s): {missing}")

    max_idx = int(folds["sample_idx"].max())
    if max_idx >= len(data):
        raise ValueError(
            f"{folds_path}: sample_idx {max_idx} is outside dataset size {len(data)}"
        )

    min_fold = int(folds["fold"].min())
    if min_fold < 1:
        raise ValueError(
            f"{folds_path}: fold numbers must start at 1; found fold {min_fold}"
        )

    duplicate_assignments = folds.duplicated(subset=["sample_idx", "fold"])
    if duplicate_assignments.any():
        duplicates = folds.loc[duplicate_assignments, ["sample_idx", "fold"]].head().to_dict("records")
        raise ValueError(f"{folds_path}: duplicated fold assignments, examples: {duplicates}")

    return data, labels, folds


def scale_features(values: np.ndarray, scaler_name: str) -> np.ndarray:
    values = values.astype(np.float32, copy=False)

    if scaler_name == "none":
        return values
    if scaler_name == "standard":
        return StandardScaler().fit_transform(values).astype(np.float32)

    return MinMaxScaler().fit_transform(values).astype(np.float32)


def load_checkpoint_state(checkpoint_path: Optional[Path]) -> Optional[Dict[str, torch.Tensor]]:
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    return checkpoint


def extract_actor_state(state: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if state is None:
        return None

    if "attention.log_alpha" in state:
        return state

    actor_state = {}
    for key, value in state.items():
        if key.startswith("actor."):
            actor_state[key.removeprefix("actor.")] = value

    if "attention.log_alpha" not in actor_state:
        keys = ", ".join(list(state.keys())[:5])
        raise ValueError(
            "Could not find HACNet actor weights in checkpoint. "
            f"First checkpoint keys: {keys}"
        )

    return actor_state


def infer_image_scale(actor_state: Optional[Dict[str, torch.Tensor]], fallback: int) -> int:
    if actor_state is None:
        return fallback

    log_alpha = actor_state["attention.log_alpha"]
    n_pixels = int(log_alpha.shape[1])
    image_scale = int(math.sqrt(n_pixels))
    if image_scale * image_scale != n_pixels:
        raise ValueError(f"Checkpoint has non-square pixel count: {n_pixels}")

    return image_scale


def build_actor(
    in_dim: int,
    image_scale: int,
    args: argparse.Namespace,
    actor_state: Optional[Dict[str, torch.Tensor]],
) -> HACNet:
    hparams = {
        "image_scale": image_scale,
        "reg_coef": args.reg_coef,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "max_iteration": args.max_iteration,
    }
    actor = HACNet(hparams, in_dim)

    if actor_state is not None:
        checkpoint_in_dim = int(actor_state["attention.log_alpha"].shape[2])
        if checkpoint_in_dim != in_dim:
            raise ValueError(
                f"Checkpoint in_dim is {checkpoint_in_dim}, but dataset in_dim is {in_dim}. "
                "Use a HACNet checkpoint trained for this dataset shape."
            )
        actor.load_state_dict(actor_state)

    actor.eval()
    return actor


def image_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    return (array * 255).round().astype(np.uint8)


def iter_batches(items: Iterable[object], batch_size: int) -> Iterable[List[object]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def image_path_for(
    output_root: Path,
    dataset_name: str,
    fold: int,
    label: object,
    sample_idx: int,
    flat: bool,
) -> Path:
    fold_dir = output_root / dataset_name / f"fold_{fold}"
    if not flat:
        fold_dir = fold_dir / f"class_{label}"
    return fold_dir / f"sample_{sample_idx:06d}.png"


def export_dataset(
    dataset_dir: Path,
    args: argparse.Namespace,
    checkpoint_state: Optional[Dict[str, torch.Tensor]],
) -> Path:
    data, labels, folds = load_dataset(dataset_dir)
    features = scale_features(data.values, args.scaler)

    actor_state = extract_actor_state(checkpoint_state)
    image_scale = infer_image_scale(actor_state, args.image_scale)
    actor = build_actor(features.shape[1], image_scale, args, actor_state).to(args.device)

    metadata_rows = []
    dataset_output = args.output_dir / dataset_dir.name
    metadata_path = dataset_output / "metadata.csv"
    dataset_output.mkdir(parents=True, exist_ok=True)

    assignments = folds[["sample_idx", "fold"]].copy()
    assignments["sample_idx"] = assignments["sample_idx"].astype(int)
    assignments["fold"] = assignments["fold"].astype(int)
    assignments = assignments.sort_values(["fold", "sample_idx"])

    assignment_records = [
        (int(row.sample_idx), int(row.fold))
        for row in assignments.itertuples(index=False)
    ]
    if args.limit is not None:
        assignment_records = assignment_records[: args.limit]

    for batch_records in iter_batches(assignment_records, args.batch_size):
        batch_indices = [sample_idx for sample_idx, _ in batch_records]
        batch = torch.from_numpy(features[batch_indices]).float().to(args.device)
        with torch.inference_mode():
            pixels, _ = actor(batch)
            images = pixels.reshape(-1, image_scale, image_scale)

        for offset, (sample_idx, fold) in enumerate(batch_records):
            label = labels.iloc[sample_idx]
            sample_id = str(data.index[sample_idx])
            out_path = image_path_for(
                args.output_dir,
                dataset_dir.name,
                fold,
                label,
                sample_idx,
                args.flat,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.overwrite or not out_path.exists():
                Image.fromarray(image_to_uint8(images[offset]), mode="L").save(out_path)

            metadata_rows.append(
                {
                    "dataset": dataset_dir.name,
                    "sample_idx": sample_idx,
                    "sample_id": sample_id,
                    "fold": fold,
                    "label": label,
                    "image_path": out_path.as_posix(),
                }
            )

    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample_idx", "sample_id", "fold", "label", "image_path"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    return metadata_path


def main() -> None:
    args = parse_args()
    checkpoint_state = load_checkpoint_state(args.checkpoint)
    dataset_dirs = discover_datasets(args.datasets_root, args.dataset)

    if args.checkpoint is None:
        print("WARNING: no checkpoint supplied; exporting images from a randomly initialized HACNet actor.")

    for dataset_dir in dataset_dirs:
        print(f"Exporting {dataset_dir.name}...")
        metadata_path = export_dataset(dataset_dir, args, checkpoint_state)
        print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
