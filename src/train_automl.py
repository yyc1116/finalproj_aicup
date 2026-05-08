"""Train AutoGluon tabular models for action, point, and rally-win prediction.

Examples:
    python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 5 --time-limit 600
    python src/train_automl.py --train-path data/train_mini.csv --model-dir models/debug_automl --k 3 --time-limit 60 --presets medium_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from features import (
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_train_features,
    get_model_feature_columns,
    make_group_split,
)

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "action": {
        "label": "target_actionId",
        "problem_type": "multiclass",
        "eval_metric": "f1_macro",
    },
    "point": {
        "label": "target_pointId",
        "problem_type": "multiclass",
        "eval_metric": "f1_macro",
    },
    "win": {
        "label": "target_serverGetPoint",
        "problem_type": "binary",
        "eval_metric": "roc_auc",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AutoGluon baselines for next-stroke and rally-win prediction.",
        epilog=(
            "Examples:\n"
            "  python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 5 --time-limit 600\n"
            "  python src/train_automl.py --train-path data/train_mini.csv --model-dir models/debug_automl --k 3 --time-limit 60 --presets medium_quality"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-path", default="data/train.csv", help="Path to train.csv or train_mini.csv")
    parser.add_argument("--model-dir", default="models/automl", help="Directory to save AutoGluon models")
    parser.add_argument("--k", type=int, default=5, help="Number of lag strokes to use")
    parser.add_argument("--time-limit", type=int, default=600, help="Per-model training time limit in seconds")
    parser.add_argument("--presets", default="medium_quality", help="AutoGluon presets")
    parser.add_argument("--valid-frac", type=float, default=0.2, help="Validation fraction for grouped split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for grouped split")
    return parser.parse_args()


def ensure_feature_columns(feature_columns: List[str]) -> None:
    overlap = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if overlap:
        raise ValueError(f"Forbidden feature columns detected: {sorted(overlap)}")


def train_one_predictor(
    name: str,
    spec: Dict[str, str],
    feature_df: pd.DataFrame,
    train_index: pd.Index,
    valid_index: pd.Index,
    feature_columns: List[str],
    model_dir: Path,
    time_limit: int,
    presets: str,
) -> None:
    from autogluon.tabular import TabularPredictor

    label = spec["label"]
    predictor_path = model_dir / name
    train_frame = feature_df.loc[train_index, feature_columns + [label]].copy()
    valid_frame = feature_df.loc[valid_index, feature_columns + [label]].copy()

    predictor = TabularPredictor(
        label=label,
        path=str(predictor_path),
        problem_type=spec["problem_type"],
        eval_metric=spec["eval_metric"],
    )
    predictor.fit(
        train_data=train_frame,
        tuning_data=valid_frame,
        presets=presets,
        time_limit=time_limit,
    )

    leaderboard = predictor.leaderboard(valid_frame, silent=True)
    leaderboard.to_csv(model_dir / f"{name}_leaderboard.csv", index=False)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    feature_df = build_train_features(train_df, k=args.k)
    feature_columns = get_model_feature_columns(feature_df)
    ensure_feature_columns(feature_columns)

    train_index, valid_index, split_info = make_group_split(
        feature_df,
        valid_frac=args.valid_frac,
        random_state=args.random_state,
    )

    if not set(TARGET_COLUMNS).issubset(feature_df.columns):
        raise ValueError(f"Missing required target columns in feature DataFrame: {TARGET_COLUMNS}")

    metadata = {
        "k": args.k,
        "train_path": args.train_path,
        "model_specs": MODEL_SPECS,
        "feature_columns": feature_columns,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "split_strategy": {
            "group_column": split_info.group_column,
            "valid_frac": args.valid_frac,
            "random_state": args.random_state,
            "n_train_rows": split_info.n_train_rows,
            "n_valid_rows": split_info.n_valid_rows,
            "n_train_groups": split_info.n_train_groups,
            "n_valid_groups": split_info.n_valid_groups,
        },
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for name, spec in MODEL_SPECS.items():
        print(f"Training {name} model with label={spec['label']} and metric={spec['eval_metric']}")
        train_one_predictor(
            name=name,
            spec=spec,
            feature_df=feature_df,
            train_index=train_index,
            valid_index=valid_index,
            feature_columns=feature_columns,
            model_dir=model_dir,
            time_limit=args.time_limit,
            presets=args.presets,
        )

    print(f"Saved models and metadata under: {model_dir}")


if __name__ == "__main__":
    main()
