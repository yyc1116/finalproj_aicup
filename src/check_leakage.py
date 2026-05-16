"""Run leakage and data-semantics checks for the cowork-style prefix feature pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd

from features import (
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_train_features,
    get_model_feature_columns,
    make_group_folds,
)

EXPECTED_FEATURE_COLUMNS = [
    "context_bucket",
    "target_stage",
    "score_state",
    "next_player_id",
    "next_player_last_actionId",
    "next_opponent_last_pointId",
    "lag1_actionId",
    "lag1_pointId",
    "lag1_actionId_transition",
    "lag2_actionId_transition",
    "action_count_0",
    "action_ratio_18",
    "point_count_0",
    "point_ratio_9",
    "serve_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check for leakage and feature-pipeline invariants.",
        epilog=(
            "Examples:\n"
            "  python src/check_leakage.py --train-path data/train.csv --k 8 --folds 5\n"
            "  python src/check_leakage.py --train-path data/train.csv --k 5 --folds 3"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-path", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--k", type=int, default=8, help="Number of lag strokes to use")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold folds to validate")
    parser.add_argument(
        "--include-match-feature",
        type=_int_flag,
        default=0,
        choices=(0, 1),
        help="Include match as a model feature",
    )
    parser.add_argument(
        "--include-rally-id-feature",
        type=_int_flag,
        default=0,
        choices=(0, 1),
        help="Include rally_id as a model feature",
    )
    return parser.parse_args()


def _int_flag(value: str) -> int:
    int_value = int(value)
    if int_value not in {0, 1}:
        raise argparse.ArgumentTypeError(f"Expected 0 or 1, got {value}")
    return int_value


def inspect_score_columns(train_df: pd.DataFrame) -> None:
    sorted_df = train_df.sort_values(["rally_uid", "strikeNumber"], kind="mergesort")
    score_self_diff = sorted_df.groupby("rally_uid")["scoreSelf"].diff().dropna()
    score_other_diff = sorted_df.groupby("rally_uid")["scoreOther"].diff().dropna()
    print("Score leakage inspection:")
    print(f"  scoreSelf negative diffs: {(score_self_diff < 0).sum()}")
    print(f"  scoreSelf diffs > 1: {(score_self_diff > 1).sum()}")
    print(f"  scoreOther negative diffs: {(score_other_diff < 0).sum()}")
    print(f"  scoreOther diffs > 1: {(score_other_diff > 1).sum()}")
    print("  Warning: score columns remain as features and should be interpreted as prefix-only context.")


def collect_errors(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    folds: List[tuple[pd.Index, pd.Index]],
    split_group_column: str,
    k: int,
    include_match_feature: bool,
    include_rally_id_feature: bool,
) -> List[str]:
    errors: List[str] = []

    for expected_column in EXPECTED_FEATURE_COLUMNS:
        if expected_column not in feature_df.columns:
            errors.append(f"Missing cowork-style feature column: {expected_column}")

    fold_valid_rallies: set[int] = set()
    fold_valid_matches: set[int] = set()
    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        train_rallies = set(feature_df.loc[train_index, "rally_uid"].tolist())
        valid_rallies = set(feature_df.loc[valid_index, "rally_uid"].tolist())
        rally_overlap = train_rallies & valid_rallies
        if rally_overlap:
            errors.append(f"Fold {fold_number} rally_uid overlap detected: {sorted(rally_overlap)[:5]}")
        if fold_valid_rallies & valid_rallies:
            errors.append(f"Fold {fold_number} validation rally_uid overlaps another fold")
        fold_valid_rallies |= valid_rallies

        if split_group_column == "match":
            train_matches = set(feature_df.loc[train_index, "match"].tolist())
            valid_matches = set(feature_df.loc[valid_index, "match"].tolist())
            match_overlap = train_matches & valid_matches
            if match_overlap:
                errors.append(f"Fold {fold_number} match overlap detected: {sorted(match_overlap)[:5]}")
            if fold_valid_matches & valid_matches:
                errors.append(f"Fold {fold_number} validation match overlaps another fold")
            fold_valid_matches |= valid_matches

    forbidden_overlap = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if forbidden_overlap:
        errors.append(f"Forbidden feature columns detected: {sorted(forbidden_overlap)}")

    if not set(TARGET_COLUMNS).issubset(feature_df.columns):
        errors.append(f"Missing target columns in feature frame: {TARGET_COLUMNS}")

    if include_match_feature and "match" not in feature_columns:
        errors.append("match should be included in feature columns when --include-match-feature=1")
    if not include_match_feature and "match" in feature_columns:
        errors.append("match should not be included in feature columns when --include-match-feature=0")
    if include_rally_id_feature and "rally_id" not in feature_columns:
        errors.append("rally_id should be included in feature columns when --include-rally-id-feature=1")
    if not include_rally_id_feature and "rally_id" in feature_columns:
        errors.append("rally_id should not be included in feature columns when --include-rally-id-feature=0")

    if not (feature_df["feature_max_strikeNumber"] < feature_df["target_strikeNumber"]).all():
        errors.append("feature_max_strikeNumber must be strictly smaller than target_strikeNumber for every row")

    for lag in range(1, k + 1):
        lag_action = f"lag{lag}_actionId"
        lag_point = f"lag{lag}_pointId"
        lag_available = f"lag{lag}_is_available"
        transition_column = f"lag{lag}_actionId_transition"
        delta_column = f"lag{lag}_actionId_delta"

        for column in (lag_action, lag_point, lag_available, transition_column, delta_column):
            if column not in feature_df.columns:
                errors.append(f"Missing lag-derived column: {column}")

        if lag_action in feature_df.columns and feature_df[lag_action].isna().any():
            errors.append(f"{lag_action} contains NaN values")
        if lag_point in feature_df.columns and feature_df[lag_point].isna().any():
            errors.append(f"{lag_point} contains NaN values")

        short_prefix_mask = feature_df["context_len"] < lag
        if lag_action in feature_df.columns and not (feature_df.loc[short_prefix_mask, lag_action].astype(int) == 0).all():
            errors.append(f"{lag_action} must equal 0 when context_len is shorter than lag {lag}")
        if lag_point in feature_df.columns and not (feature_df.loc[short_prefix_mask, lag_point].astype(int) == 0).all():
            errors.append(f"{lag_point} must equal 0 when context_len is shorter than lag {lag}")
        if lag_available in feature_df.columns and not (feature_df.loc[short_prefix_mask, lag_available].astype(int) == 0).all():
            errors.append(f"{lag_available} must equal 0 when context_len is shorter than lag {lag}")

        missing_transition_mask = feature_df["context_len"] < (lag + 1)
        if transition_column in feature_df.columns and not (
            feature_df.loc[missing_transition_mask, transition_column].astype(int) == 0
        ).all():
            errors.append(f"{transition_column} must equal 0 when the transition history is unavailable")
        if delta_column in feature_df.columns and not (feature_df.loc[missing_transition_mask, delta_column].astype(int) == 0).all():
            errors.append(f"{delta_column} must equal 0 when the transition history is unavailable")

    raw_action_zeros = int((train_df["actionId"] == 0).sum())
    raw_point_zeros = int((train_df["pointId"] == 0).sum())
    target_action_zeros = int((feature_df["target_actionId"] == 0).sum())
    target_point_zeros = int((feature_df["target_pointId"] == 0).sum())
    print("Zero-class counts:")
    print(f"  raw actionId == 0: {raw_action_zeros}")
    print(f"  raw pointId == 0: {raw_point_zeros}")
    print(f"  target_actionId == 0: {target_action_zeros}")
    print(f"  target_pointId == 0: {target_point_zeros}")

    if raw_action_zeros <= 0 or target_action_zeros <= 0:
        errors.append("actionId = 0 should remain present in raw data and training targets")
    if raw_point_zeros <= 0 or target_point_zeros <= 0:
        errors.append("pointId = 0 should remain present in raw data and training targets")

    return errors


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)
    feature_df = build_train_features(train_df, k=args.k, cast_categoricals=False)
    feature_columns = get_model_feature_columns(
        feature_df,
        include_match_feature=bool(args.include_match_feature),
        include_rally_id_feature=bool(args.include_rally_id_feature),
    )
    folds, fold_info = make_group_folds(feature_df, n_splits=args.folds)

    errors = collect_errors(
        train_df=train_df,
        feature_df=feature_df,
        feature_columns=feature_columns,
        folds=folds,
        split_group_column=fold_info.group_column,
        k=args.k,
        include_match_feature=bool(args.include_match_feature),
        include_rally_id_feature=bool(args.include_rally_id_feature),
    )

    print("Split summary:")
    print(f"  group column: {fold_info.group_column}")
    print(f"  folds: {fold_info.n_splits}")
    print(f"  rows: {fold_info.n_rows}")
    print(f"  groups: {fold_info.n_groups}")
    inspect_score_columns(train_df)

    if errors:
        print("Leakage checks failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("All hard leakage checks passed.")


if __name__ == "__main__":
    main()
