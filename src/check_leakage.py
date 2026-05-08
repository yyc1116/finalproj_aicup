"""Run leakage and data-semantics checks for the prefix feature pipeline.

Examples:
    python src/check_leakage.py --train-path data/train.csv --k 5
    python src/check_leakage.py --train-path data/train_mini.csv --k 3
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd

from features import (
    FORBIDDEN_FEATURE_COLUMNS,
    build_train_features,
    get_model_feature_columns,
    make_group_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check for leakage and feature-pipeline invariants.",
        epilog=(
            "Examples:\n"
            "  python src/check_leakage.py --train-path data/train.csv --k 5\n"
            "  python src/check_leakage.py --train-path data/train_mini.csv --k 3"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-path", default="data/train.csv", help="Path to train.csv or train_mini.csv")
    parser.add_argument("--k", type=int, default=5, help="Number of lag strokes to use")
    parser.add_argument("--valid-frac", type=float, default=0.2, help="Validation fraction for grouped split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for grouped split")
    return parser.parse_args()


def inspect_score_columns(train_df: pd.DataFrame) -> None:
    sorted_df = train_df.sort_values(["rally_uid", "strikeNumber"], kind="mergesort")
    score_self_diff = sorted_df.groupby("rally_uid")["scoreSelf"].diff().dropna()
    score_other_diff = sorted_df.groupby("rally_uid")["scoreOther"].diff().dropna()
    print("Score leakage inspection:")
    print(f"  scoreSelf negative diffs: {(score_self_diff < 0).sum()}")
    print(f"  scoreSelf diffs > 1: {(score_self_diff > 1).sum()}")
    print(f"  scoreOther negative diffs: {(score_other_diff < 0).sum()}")
    print(f"  scoreOther diffs > 1: {(score_other_diff > 1).sum()}")
    print("  Warning: score columns are retained as requested features, but their within-rally behavior should be reviewed.")


def collect_errors(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    train_index: pd.Index,
    valid_index: pd.Index,
    split_group_column: str,
    k: int,
) -> List[str]:
    errors: List[str] = []

    train_rallies = set(feature_df.loc[train_index, "rally_uid"].tolist())
    valid_rallies = set(feature_df.loc[valid_index, "rally_uid"].tolist())
    rally_overlap = train_rallies & valid_rallies
    if rally_overlap:
        errors.append(f"Train/valid rally_uid overlap detected: {sorted(rally_overlap)[:5]}")

    if split_group_column == "match":
        train_matches = set(feature_df.loc[train_index, "match"].tolist())
        valid_matches = set(feature_df.loc[valid_index, "match"].tolist())
        match_overlap = train_matches & valid_matches
        if match_overlap:
            errors.append(f"Train/valid match overlap detected: {sorted(match_overlap)[:5]}")

    forbidden_overlap = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if forbidden_overlap:
        errors.append(f"Forbidden feature columns detected: {sorted(forbidden_overlap)}")

    if not (feature_df["feature_max_strikeNumber"] < feature_df["target_strikeNumber"]).all():
        errors.append("feature_max_strikeNumber must be strictly smaller than target_strikeNumber for every row")

    for lag in range(1, k + 1):
        for suffix in ("actionId", "pointId"):
            column = f"prev{lag}_{suffix}"
            if feature_df[column].isna().any():
                errors.append(f"{column} contains NaN values; missing lag values must be zero-filled")
            short_prefix_mask = feature_df["rally_len_so_far"] < lag
            if not (feature_df.loc[short_prefix_mask, column] == 0).all():
                errors.append(f"{column} must equal 0 when the prefix length is shorter than lag {lag}")

    raw_action_zeros = int((train_df["actionId"] == 0).sum())
    raw_point_zeros = int((train_df["pointId"] == 0).sum())
    target_action_zeros = int((feature_df["target_actionId"] == 0).sum())
    target_point_zeros = int((feature_df["target_pointId"] == 0).sum())
    print("Zero-class counts:")
    print(f"  raw actionId == 0: {raw_action_zeros}")
    print(f"  raw pointId == 0: {raw_point_zeros}")
    print(f"  target_actionId == 0: {target_action_zeros}")
    print(f"  target_pointId == 0: {target_point_zeros}")

    return errors


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)
    feature_df = build_train_features(train_df, k=args.k)
    feature_columns = get_model_feature_columns(feature_df)
    train_index, valid_index, split_info = make_group_split(
        feature_df,
        valid_frac=args.valid_frac,
        random_state=args.random_state,
    )

    errors = collect_errors(
        train_df=train_df,
        feature_df=feature_df,
        feature_columns=feature_columns,
        train_index=train_index,
        valid_index=valid_index,
        split_group_column=split_info.group_column,
        k=args.k,
    )

    print("Split summary:")
    print(f"  group column: {split_info.group_column}")
    print(f"  train rows: {split_info.n_train_rows}")
    print(f"  valid rows: {split_info.n_valid_rows}")
    print(f"  train groups: {split_info.n_train_groups}")
    print(f"  valid groups: {split_info.n_valid_groups}")
    inspect_score_columns(train_df)

    if errors:
        print("Leakage checks failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("All hard leakage checks passed.")


if __name__ == "__main__":
    main()
