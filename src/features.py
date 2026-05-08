"""Shared feature engineering utilities for the table-tennis AutoML baseline.

Example usage:
    python src/check_leakage.py --train-path data/train.csv --k 5
    python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 5 --time-limit 600
    python src/predict_submission.py --test-path data/test_new.csv --model-dir models/automl --k 5 --out-path submission.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_CONTEXT_COLUMNS = ["sex", "match", "numberGame", "rally_id"]
RAW_STATE_COLUMNS = ["scoreSelf", "scoreOther"]
LAG_SOURCE_COLUMNS = [
    "actionId",
    "pointId",
    "spinId",
    "strengthId",
    "handId",
    "strikeId",
    "positionId",
    "gamePlayerId",
    "gamePlayerOtherId",
]
TARGET_COLUMNS = ["target_actionId", "target_pointId", "target_serverGetPoint"]
FORBIDDEN_FEATURE_COLUMNS = set(TARGET_COLUMNS + ["serverGetPoint"])
NON_MODEL_FEATURE_COLUMNS = {"rally_uid", "feature_max_strikeNumber", "target_strikeNumber"}
TRAIN_REQUIRED_COLUMNS = set(
    ["rally_uid", "strikeNumber", "serverGetPoint"]
    + RAW_CONTEXT_COLUMNS
    + RAW_STATE_COLUMNS
    + LAG_SOURCE_COLUMNS
)
TEST_REQUIRED_COLUMNS = TRAIN_REQUIRED_COLUMNS - {"serverGetPoint"}


@dataclass(frozen=True)
class SplitInfo:
    group_column: str
    n_train_rows: int
    n_valid_rows: int
    n_train_groups: int
    n_valid_groups: int


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def get_model_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in feature_df.columns
        if column not in FORBIDDEN_FEATURE_COLUMNS and column not in NON_MODEL_FEATURE_COLUMNS
    ]


def make_group_split(
    feature_df: pd.DataFrame,
    valid_frac: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Index, pd.Index, SplitInfo]:
    if not 0 < valid_frac < 1:
        raise ValueError(f"valid_frac must be between 0 and 1, got {valid_frac}")
    if feature_df.empty:
        raise ValueError("Cannot create a validation split from an empty feature DataFrame")

    group_column = "rally_uid"
    if "match" in feature_df.columns:
        match_series = feature_df["match"]
        if match_series.notna().all() and match_series.nunique() > 1:
            group_column = "match"

    groups = feature_df[group_column]
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_frac, random_state=random_state)
    train_pos, valid_pos = next(splitter.split(feature_df, groups=groups))
    train_index = feature_df.index.take(train_pos)
    valid_index = feature_df.index.take(valid_pos)

    train_groups = set(feature_df.loc[train_index, group_column].tolist())
    valid_groups = set(feature_df.loc[valid_index, group_column].tolist())
    overlap = train_groups & valid_groups
    if overlap:
        sample_overlap = sorted(overlap)[:5]
        raise ValueError(f"Group split leakage detected on {group_column}: {sample_overlap}")

    return train_index, valid_index, SplitInfo(
        group_column=group_column,
        n_train_rows=len(train_index),
        n_valid_rows=len(valid_index),
        n_train_groups=len(train_groups),
        n_valid_groups=len(valid_groups),
    )


def build_train_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Build prefix-based training rows that predict the next stroke."""
    _validate_k(k)
    validate_required_columns(df, TRAIN_REQUIRED_COLUMNS, "train DataFrame")
    sorted_df = _sort_strokes(df)

    rows: List[Dict[str, Any]] = []
    for rally_uid, rally_df in sorted_df.groupby("rally_uid", sort=False):
        rows.extend(_build_train_rows_for_rally(rally_uid, rally_df, k))

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No training samples were produced. Check the input training data.")
    return feature_df


def build_test_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Build one prefix-based inference row per rally."""
    _validate_k(k)
    validate_required_columns(df, TEST_REQUIRED_COLUMNS, "test DataFrame")

    rally_order = df["rally_uid"].drop_duplicates().tolist()
    sorted_df = _sort_strokes(df)
    grouped = {rally_uid: rally_df for rally_uid, rally_df in sorted_df.groupby("rally_uid", sort=False)}

    rows: List[Dict[str, Any]] = []
    for rally_uid in rally_order:
        rally_df = grouped[rally_uid]
        prefix = rally_df.reset_index(drop=True)
        rows.append(_build_prefix_row(prefix=prefix, k=k, include_targets=False))

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No test samples were produced. Check the input test data.")
    return feature_df


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def _sort_strokes(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").reset_index(drop=True)


def _build_train_rows_for_rally(rally_uid: Any, rally_df: pd.DataFrame, k: int) -> List[Dict[str, Any]]:
    prefix_rows: List[Dict[str, Any]] = []
    rally_df = rally_df.reset_index(drop=True)
    if len(rally_df) < 2:
        return prefix_rows

    label_values = rally_df["serverGetPoint"].dropna().unique().tolist()
    if len(label_values) != 1:
        raise ValueError(f"serverGetPoint must be constant within rally {rally_uid}, got {label_values}")

    for current_end in range(len(rally_df) - 1):
        prefix = rally_df.iloc[: current_end + 1].reset_index(drop=True)
        next_row = rally_df.iloc[current_end + 1]
        row = _build_prefix_row(prefix=prefix, k=k, include_targets=True)
        row["target_actionId"] = int(next_row["actionId"])
        row["target_pointId"] = int(next_row["pointId"])
        row["target_serverGetPoint"] = int(label_values[0])
        row["target_strikeNumber"] = int(next_row["strikeNumber"])
        prefix_rows.append(row)

    return prefix_rows


def _build_prefix_row(prefix: pd.DataFrame, k: int, include_targets: bool) -> Dict[str, Any]:
    last_row = prefix.iloc[-1]
    tail = prefix.tail(k)
    row: Dict[str, Any] = {
        "rally_uid": last_row["rally_uid"],
        "sex": int(last_row["sex"]),
        "match": int(last_row["match"]),
        "numberGame": int(last_row["numberGame"]),
        "rally_id": int(last_row["rally_id"]),
        "current_strikeNumber": int(last_row["strikeNumber"]),
        "current_scoreSelf": int(last_row["scoreSelf"]),
        "current_scoreOther": int(last_row["scoreOther"]),
        "rally_len_so_far": int(len(prefix)),
        "feature_max_strikeNumber": int(last_row["strikeNumber"]),
        "tail_k_action_nunique": int(tail["actionId"].nunique(dropna=False)),
        "tail_k_point_nunique": int(tail["pointId"].nunique(dropna=False)),
        "tail_k_action_zero_count": int((tail["actionId"] == 0).sum()),
        "tail_k_point_zero_count": int((tail["pointId"] == 0).sum()),
        "tail_k_last_strengthId": int(last_row["strengthId"]),
        "tail_k_last_gamePlayerId": int(last_row["gamePlayerId"]),
        "tail_k_last_gamePlayerOtherId": int(last_row["gamePlayerOtherId"]),
    }

    for lag in range(1, k + 1):
        if lag <= len(prefix):
            source_row = prefix.iloc[-lag]
            for column in LAG_SOURCE_COLUMNS:
                row[f"prev{lag}_{column}"] = int(source_row[column])
        else:
            for column in LAG_SOURCE_COLUMNS:
                row[f"prev{lag}_{column}"] = 0

    if include_targets:
        row["target_strikeNumber"] = 0

    return row
