"""Train AutoGluon tabular models for action, point, and rally-win prediction.

Examples:
    python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 5 --time-limit 600
    python src/train_automl.py --train-path data/train_mini.csv --model-dir models/debug_automl --k 3 --time-limit 60 --presets medium_quality
    python src/train_automl.py --train-path data/train.csv --model-dir models/automl_best --k 7 --time-limit 3600 --presets best_quality --num-bag-folds 5 --num-stack-levels 1 --refit-full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
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


def maybe_tqdm(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AutoGluon baselines for next-stroke and rally-win prediction.",
        epilog=(
            "Examples:\n"
            "  python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 5 --time-limit 600\n"
            "  python src/train_automl.py --train-path data/train_mini.csv --model-dir models/debug_automl --k 3 --time-limit 60 --presets medium_quality\n"
            "  python src/train_automl.py --train-path data/train.csv --model-dir models/automl_best --k 7 --time-limit 3600 --presets best_quality --num-bag-folds 5 --num-stack-levels 1 --refit-full"
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
    parser.add_argument("--num-bag-folds", type=int, default=0, help="AutoGluon bagging folds; use 5 for stronger ensembles")
    parser.add_argument("--num-bag-sets", type=int, default=1, help="AutoGluon bagging repeats when bagging is enabled")
    parser.add_argument("--num-stack-levels", type=int, default=0, help="AutoGluon stack levels; use 1 for stronger ensembles")
    parser.add_argument("--refit-full", action="store_true", help="Refit the best model on train+valid after selection")
    parser.add_argument("--keep-only-best", action="store_true", help="Delete non-best submodels after training to save space")
    parser.add_argument("--save-space", action="store_true", help="Ask AutoGluon to minimize model artifact size")
    parser.add_argument("--ag-verbosity", type=int, default=2, help="AutoGluon verbosity level")
    return parser.parse_args()


def ensure_feature_columns(feature_columns: List[str]) -> None:
    overlap = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if overlap:
        raise ValueError(f"Forbidden feature columns detected: {sorted(overlap)}")


def build_fit_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    fit_kwargs: Dict[str, object] = {
        "presets": args.presets,
        "time_limit": args.time_limit,
        "verbosity": args.ag_verbosity,
    }
    if args.num_bag_folds > 0:
        fit_kwargs["num_bag_folds"] = args.num_bag_folds
        fit_kwargs["num_bag_sets"] = args.num_bag_sets
    if args.num_stack_levels > 0:
        fit_kwargs["num_stack_levels"] = args.num_stack_levels
    if args.refit_full:
        fit_kwargs["refit_full"] = True
    if args.keep_only_best:
        fit_kwargs["keep_only_best"] = True
    if args.save_space:
        fit_kwargs["save_space"] = True
    return fit_kwargs


def train_one_predictor(
    name: str,
    spec: Dict[str, str],
    feature_df: pd.DataFrame,
    train_index: pd.Index,
    valid_index: pd.Index,
    feature_columns: List[str],
    model_dir: Path,
    fit_kwargs: Dict[str, object],
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
    predictor.fit(train_data=train_frame, tuning_data=valid_frame, **fit_kwargs)

    leaderboard = predictor.leaderboard(valid_frame, silent=True)
    leaderboard.to_csv(model_dir / f"{name}_leaderboard.csv", index=False)


def main() -> None:
    args = parse_args()
    fit_kwargs = build_fit_kwargs(args)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()
    print(f"[1/5] Loading training data from {args.train_path}")
    train_df = pd.read_csv(args.train_path)
    print(f"Loaded {len(train_df):,} stroke rows")

    print(f"[2/5] Building prefix features with k={args.k}")
    feature_df = build_train_features(train_df, k=args.k)
    feature_columns = get_model_feature_columns(feature_df)
    ensure_feature_columns(feature_columns)
    print(f"Built {len(feature_df):,} training samples with {len(feature_columns):,} model features")

    print(f"[3/5] Creating grouped validation split")
    train_index, valid_index, split_info = make_group_split(
        feature_df,
        valid_frac=args.valid_frac,
        random_state=args.random_state,
    )

    if not set(TARGET_COLUMNS).issubset(feature_df.columns):
        raise ValueError(f"Missing required target columns in feature DataFrame: {TARGET_COLUMNS}")

    print(
        f"Split by {split_info.group_column}: "
        f"{split_info.n_train_rows:,} train rows / {split_info.n_valid_rows:,} valid rows"
    )

    print(f"[4/5] Saving training metadata to {model_dir}")
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
        "fit_kwargs": fit_kwargs,
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[5/5] Training AutoGluon predictors")
    model_items = list(MODEL_SPECS.items())
    for step, (name, spec) in enumerate(maybe_tqdm(model_items, total=len(model_items), desc="Models"), start=1):
        model_start = time.perf_counter()
        print(
            f"Starting model {step}/{len(model_items)}: {name} "
            f"(label={spec['label']}, metric={spec['eval_metric']})"
        )
        train_one_predictor(
            name=name,
            spec=spec,
            feature_df=feature_df,
            train_index=train_index,
            valid_index=valid_index,
            feature_columns=feature_columns,
            model_dir=model_dir,
            fit_kwargs=fit_kwargs,
        )
        elapsed = time.perf_counter() - model_start
        print(f"Finished {name} in {elapsed / 60:.1f} minutes")

    total_elapsed = time.perf_counter() - overall_start
    print(f"Saved models and metadata under: {model_dir}")
    print(f"Total elapsed time: {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
