"""Shared cowork-style preprocessing and prefix feature utilities.

Example usage:
    python src/check_leakage.py --train-path data/train.csv --k 8 --folds 5
    python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 8 --folds 5
    python src/predict_submission.py --test-path data/test_new.csv --model-dir models/automl --k 8 --out-path submission.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PREPROCESSING_SIGNATURE = "cowork_catboost_prefix_v1_zero_pad"

ACTION_CLASSES = list(range(19))
POINT_CLASSES = list(range(10))
RAW_CONTEXT_COLUMNS = ["sex", "match", "numberGame", "rally_id"]
RAW_STATE_COLUMNS = ["scoreSelf", "scoreOther"]
RAW_PLAYER_COLUMNS = ["gamePlayerId", "gamePlayerOtherId"]
RAW_STROKE_COLUMNS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
TRAIN_REQUIRED_COLUMNS = set(
    ["rally_uid", "strikeNumber", "serverGetPoint"] + RAW_CONTEXT_COLUMNS + RAW_STATE_COLUMNS + RAW_PLAYER_COLUMNS + RAW_STROKE_COLUMNS
)
TEST_REQUIRED_COLUMNS = TRAIN_REQUIRED_COLUMNS - {"serverGetPoint"}
TARGET_COLUMNS = ["target_actionId", "target_pointId", "target_serverGetPoint"]
FORBIDDEN_FEATURE_COLUMNS = set(TARGET_COLUMNS + ["serverGetPoint"])
ALWAYS_NON_MODEL_FEATURE_COLUMNS = {"rally_uid", "feature_max_strikeNumber", "target_strikeNumber", "is_last_target"}
OPTIONAL_CONTEXT_FEATURE_COLUMNS = {
    "match": "include_match_feature",
    "rally_id": "include_rally_id_feature",
}

PAD_CAT = 0
UNKNOWN_PLAYER = 0
PAIR_BASE = 100_000

LAST_K_STROKE_COLS = [
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "gamePlayerId",
    "gamePlayerOtherId",
]
PLAYER_ID_COLS = ["gamePlayerId", "gamePlayerOtherId"]
AGG_COLS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
TRANSITION_COLS = ["strikeId", "pointId", "actionId", "positionId"]
ROLE_HISTORY_COLS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]

POINT_ROW_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
POINT_COL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}


@dataclass(frozen=True)
class GroupFoldInfo:
    group_column: str
    n_splits: int
    n_rows: int
    n_groups: int


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def prepare_raw_frame(df: pd.DataFrame, require_server_target: bool) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = prepared.columns.str.strip()
    required_columns = TRAIN_REQUIRED_COLUMNS if require_server_target else TEST_REQUIRED_COLUMNS
    validate_required_columns(prepared, required_columns, "input DataFrame")

    if "serverGetPoint" not in prepared.columns:
        prepared["serverGetPoint"] = -1

    numeric_columns = sorted(required_columns)
    if "serverGetPoint" not in numeric_columns:
        numeric_columns.append("serverGetPoint")
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(-1).astype(int)

    prepared = prepared.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").reset_index(drop=True)
    prepared["strikeNumber"] = prepared["strikeNumber"].clip(lower=0, upper=60)
    prepared["scoreDiff"] = prepared["scoreSelf"] - prepared["scoreOther"]
    prepared["scoreTotal"] = prepared["scoreSelf"] + prepared["scoreOther"]
    prepared["isServe"] = (prepared["strikeId"] == 1).astype(int)
    prepared["isReceive"] = (prepared["strikeId"] == 2).astype(int)
    prepared["isRally"] = (prepared["strikeId"] == 4).astype(int)
    prepared["isDeuce"] = ((prepared["scoreSelf"] >= 10) & (prepared["scoreOther"] >= 10)).astype(int)
    prepared["isGamePointSelf"] = (
        (prepared["scoreSelf"] >= 10) & (prepared["scoreSelf"] > prepared["scoreOther"])
    ).astype(int)
    prepared["isGamePointOpp"] = (
        (prepared["scoreOther"] >= 10) & (prepared["scoreOther"] > prepared["scoreSelf"])
    ).astype(int)
    prepared["pointIdRow"] = prepared["pointId"].map(POINT_ROW_MAP).fillna(0).astype(int)
    prepared["pointIdCol"] = prepared["pointId"].map(POINT_COL_MAP).fillna(0).astype(int)
    return prepared


def get_non_model_feature_columns(
    include_match_feature: bool = False,
    include_rally_id_feature: bool = False,
) -> set[str]:
    non_model_columns = set(ALWAYS_NON_MODEL_FEATURE_COLUMNS)
    if not include_match_feature:
        non_model_columns.add("match")
    if not include_rally_id_feature:
        non_model_columns.add("rally_id")
    return non_model_columns


def get_model_feature_columns(
    feature_df: pd.DataFrame,
    include_match_feature: bool = False,
    include_rally_id_feature: bool = False,
) -> List[str]:
    non_model_feature_columns = get_non_model_feature_columns(
        include_match_feature=include_match_feature,
        include_rally_id_feature=include_rally_id_feature,
    )
    return [
        column
        for column in feature_df.columns
        if column not in FORBIDDEN_FEATURE_COLUMNS and column not in non_model_feature_columns
    ]


def get_categorical_feature_columns(columns: Sequence[str]) -> List[str]:
    categorical_columns: List[str] = []
    seen: set[str] = set()
    for column in columns:
        if _is_categorical_feature_column(column) and column not in seen:
            categorical_columns.append(column)
            seen.add(column)
    return categorical_columns


def apply_categorical_dtypes(
    feature_df: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    cast_df = feature_df.copy()
    columns_to_cast = get_categorical_feature_columns(cast_df.columns) if categorical_columns is None else list(categorical_columns)
    for column in columns_to_cast:
        if column in cast_df.columns:
            cast_df[column] = cast_df[column].astype("category")
    return cast_df


def build_train_features(df: pd.DataFrame, k: int, cast_categoricals: bool = True) -> pd.DataFrame:
    _validate_k(k)
    prepared = prepare_raw_frame(df, require_server_target=True)

    rows: List[Dict[str, Any]] = []
    for rally_uid, group in prepared.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber", kind="mergesort").reset_index(drop=True)
        if len(group) < 2:
            continue

        label_values = group["serverGetPoint"].dropna().unique().tolist()
        if len(label_values) != 1:
            raise ValueError(f"serverGetPoint must be constant within rally {rally_uid}, got {label_values}")

        for target_pos in range(1, len(group)):
            row = _build_prefix_row(int(rally_uid), group, target_pos=target_pos, mode="train", k=k)
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No training samples were produced. Check the input training data.")
    if cast_categoricals:
        feature_df = apply_categorical_dtypes(feature_df)
    return feature_df


def build_test_features(df: pd.DataFrame, k: int, cast_categoricals: bool = True) -> pd.DataFrame:
    _validate_k(k)
    rally_order = df["rally_uid"].drop_duplicates().tolist()
    prepared = prepare_raw_frame(df, require_server_target=False)
    grouped = {rally_uid: group.reset_index(drop=True) for rally_uid, group in prepared.groupby("rally_uid", sort=False)}

    rows: List[Dict[str, Any]] = []
    for rally_uid in rally_order:
        group = grouped[rally_uid]
        rows.append(_build_prefix_row(int(rally_uid), group, target_pos=len(group), mode="test", k=k))

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No test samples were produced. Check the input test data.")
    if cast_categoricals:
        feature_df = apply_categorical_dtypes(feature_df)
    return feature_df


def choose_group_column(feature_df: pd.DataFrame, min_groups: int = 2) -> str:
    if "match" in feature_df.columns:
        match_series = feature_df["match"]
        if match_series.notna().all() and match_series.nunique() >= min_groups:
            return "match"
    return "rally_uid"


def make_group_folds(
    feature_df: pd.DataFrame,
    n_splits: int,
) -> Tuple[List[Tuple[pd.Index, pd.Index]], GroupFoldInfo]:
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if feature_df.empty:
        raise ValueError("Cannot create group folds from an empty feature DataFrame")

    group_column = choose_group_column(feature_df, min_groups=n_splits)
    groups = feature_df[group_column].to_numpy()
    unique_groups = pd.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups for GroupKFold on {group_column}, got {len(unique_groups)}")

    splitter = GroupKFold(n_splits=n_splits)
    folds: List[Tuple[pd.Index, pd.Index]] = []
    for train_pos, valid_pos in splitter.split(feature_df, groups=groups):
        train_index = feature_df.index.take(train_pos)
        valid_index = feature_df.index.take(valid_pos)
        folds.append((train_index, valid_index))

    return folds, GroupFoldInfo(
        group_column=group_column,
        n_splits=n_splits,
        n_rows=len(feature_df),
        n_groups=len(unique_groups),
    )


def player_feature_columns(k: int) -> List[str]:
    columns = ["next_player_id", "next_opponent_id"]
    for lag in range(1, k + 1):
        columns.extend([f"lag{lag}_gamePlayerId", f"lag{lag}_gamePlayerOtherId"])
    return columns


def collect_known_players(frame: pd.DataFrame, k: int) -> set[int]:
    values: List[int] = []
    for column in player_feature_columns(k):
        if column in frame.columns:
            values.extend(frame.loc[frame[column] > 0, column].astype(int).tolist())
    return set(values)


def apply_unknown_player_mapping(frame: pd.DataFrame, known_players: Iterable[int], k: int) -> pd.DataFrame:
    known_player_set = {int(player_id) for player_id in known_players if int(player_id) > 0}
    mapped = frame.copy()

    for column in player_feature_columns(k):
        if column not in mapped.columns:
            continue
        values = mapped[column].astype(int)
        mapped[column] = values.where(values.isin(known_player_set) | (values <= 0), UNKNOWN_PLAYER).astype(int)

    mapped["player_pair_id"] = [
        pair_code(a, b)
        for a, b in zip(mapped["next_player_id"].astype(int), mapped["next_opponent_id"].astype(int), strict=True)
    ]
    mapped["player_pair_unordered_id"] = [
        pair_code(min(a, b), max(a, b))
        for a, b in zip(mapped["next_player_id"].astype(int), mapped["next_opponent_id"].astype(int), strict=True)
    ]

    for lag in range(1, k + 1):
        player_col = f"lag{lag}_gamePlayerId"
        opponent_col = f"lag{lag}_gamePlayerOtherId"
        pair_col = f"lag{lag}_player_pair_id"
        if player_col not in mapped.columns or opponent_col not in mapped.columns:
            continue
        mapped[pair_col] = [
            pair_code(a, b)
            for a, b in zip(mapped[player_col].astype(int), mapped[opponent_col].astype(int), strict=True)
        ]

    return mapped


def pair_code(left_value: int, right_value: int) -> int:
    return int(left_value) * PAIR_BASE + int(right_value)


def transition_code(prev_value: int, cur_value: int, base: int = 100) -> int:
    return int(prev_value) * base + int(cur_value)


def context_bucket(length: int) -> int:
    if length <= 1:
        return 1
    if length == 2:
        return 2
    if length == 3:
        return 3
    return 4


def target_stage(strike_number: int) -> int:
    if strike_number <= 2:
        return 2
    if strike_number == 3:
        return 3
    if strike_number == 4:
        return 4
    if strike_number <= 6:
        return 5
    return 7


def score_state(score_self: int, score_other: int) -> int:
    if score_self >= 10 and score_other >= 10:
        return 4
    if score_self >= 10 and score_self > score_other:
        return 3
    if score_other >= 10 and score_other > score_self:
        return 2
    if score_self == score_other:
        return 1
    return 0


def align_feature_columns(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    aligned = feature_df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    aligned = aligned.loc[:, list(feature_columns)]
    if categorical_feature_columns:
        aligned = apply_categorical_dtypes(aligned, categorical_feature_columns)
    return aligned


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def _build_prefix_row(
    rally_uid: int,
    group: pd.DataFrame,
    target_pos: int,
    mode: str,
    k: int,
) -> Dict[str, float | int]:
    context = group.iloc[:target_pos].reset_index(drop=True)
    if context.empty:
        raise ValueError(f"Cannot build a prefix row for empty context in rally {rally_uid}")

    last = context.iloc[-1]
    next_target_strike_number = (
        int(group.iloc[target_pos]["strikeNumber"]) if mode == "train" else int(last["strikeNumber"]) + 1
    )
    context_len = len(context)
    next_player_id = int(last["gamePlayerOtherId"])
    next_opponent_id = int(last["gamePlayerId"])
    score_self = int(last["scoreSelf"])
    score_other = int(last["scoreOther"])
    score_diff = int(last["scoreDiff"])
    score_total = int(last["scoreTotal"])

    row: Dict[str, float | int] = {
        "rally_uid": int(rally_uid),
        "match": int(last["match"]),
        "rally_id": int(last["rally_id"]),
        "feature_max_strikeNumber": int(last["strikeNumber"]),
        "target_strikeNumber": int(next_target_strike_number),
        "is_last_target": int(target_pos == len(group) - 1) if mode == "train" else 0,
        "context_len": int(context_len),
        "rally_len": int(len(group)),
        "context_bucket": context_bucket(context_len),
        "target_stage": target_stage(next_target_strike_number),
        "next_strikeId": 2 if context_len == 1 else 4,
        "scoreSelf": score_self,
        "scoreOther": score_other,
        "scoreDiff": score_diff,
        "scoreTotal": score_total,
        "absScoreDiff": abs(score_diff),
        "score_state": score_state(score_self, score_other),
        "lead_state": int(np.sign(score_diff)),
        "is_tied": int(score_diff == 0),
        "is_deuce_like": int(score_self >= 10 and score_other >= 10),
        "sex": int(last["sex"]),
        "numberGame": int(last["numberGame"]),
        "next_player_id": next_player_id,
        "next_opponent_id": next_opponent_id,
        "player_pair_id": pair_code(next_player_id, next_opponent_id),
        "player_pair_unordered_id": pair_code(min(next_player_id, next_opponent_id), max(next_player_id, next_opponent_id)),
    }

    _add_role_history(row, context, "next_player", next_player_id)
    _add_role_history(row, context, "next_opponent", next_opponent_id)
    _add_lag_features(row, context, k)
    _add_prefix_aggregates(row, context)

    if mode == "train":
        target = group.iloc[target_pos]
        row["target_actionId"] = int(target["actionId"])
        row["target_pointId"] = int(target["pointId"])
        row["target_serverGetPoint"] = int(group["serverGetPoint"].iloc[0])

    return row


def _add_role_history(
    row: Dict[str, float | int],
    context: pd.DataFrame,
    role_name: str,
    player_id: int,
) -> None:
    player_rows = context[context["gamePlayerId"].astype(int) == int(player_id)]
    row[f"{role_name}_context_count"] = int(len(player_rows))
    row[f"{role_name}_context_ratio"] = float(len(player_rows) / max(len(context), 1))
    for column in ROLE_HISTORY_COLS:
        row[f"{role_name}_last_{column}"] = int(player_rows[column].iloc[-1]) if len(player_rows) else PAD_CAT


def _add_lag_features(row: Dict[str, float | int], context: pd.DataFrame, k: int) -> None:
    context_len = len(context)
    for lag in range(1, k + 1):
        has_lag = context_len >= lag
        row[f"lag{lag}_is_available"] = int(has_lag)
        if has_lag:
            stroke = context.iloc[-lag]
            for column in LAST_K_STROKE_COLS:
                row[f"lag{lag}_{column}"] = int(stroke[column])
            row[f"lag{lag}_player_pair_id"] = pair_code(int(stroke["gamePlayerId"]), int(stroke["gamePlayerOtherId"]))
        else:
            for column in LAST_K_STROKE_COLS:
                row[f"lag{lag}_{column}"] = PAD_CAT
            row[f"lag{lag}_player_pair_id"] = pair_code(UNKNOWN_PLAYER, UNKNOWN_PLAYER)

        has_transition = context_len >= lag + 1
        for column in TRANSITION_COLS:
            if has_transition:
                prev_value = int(context.iloc[-lag - 1][column])
                cur_value = int(context.iloc[-lag][column])
                row[f"lag{lag}_{column}_transition"] = transition_code(prev_value, cur_value)
                row[f"lag{lag}_{column}_delta"] = int(cur_value - prev_value)
            else:
                row[f"lag{lag}_{column}_transition"] = PAD_CAT
                row[f"lag{lag}_{column}_delta"] = 0


def _add_prefix_aggregates(row: Dict[str, float | int], context: pd.DataFrame) -> None:
    context_len = len(context)
    for column in AGG_COLS:
        values = context[column].astype(int).to_numpy()
        row[f"{column}_first"] = int(values[0])
        row[f"{column}_last"] = int(values[-1])
        row[f"{column}_mode"] = _mode_int(values)
        row[f"{column}_mean"] = float(values.mean())
        row[f"{column}_std"] = float(values.std()) if context_len >= 2 else 0.0
        row[f"{column}_nunique"] = int(pd.Series(values).nunique())
        row[f"{column}_last_delta"] = int(values[-1] - values[-2]) if context_len >= 2 else 0
        row[f"{column}_change_count"] = int((values[1:] != values[:-1]).sum()) if context_len >= 2 else 0

    for action_class in ACTION_CLASSES:
        count = int((context["actionId"].astype(int) == action_class).sum())
        row[f"action_count_{action_class}"] = count
        row[f"action_ratio_{action_class}"] = float(count / max(context_len, 1))

    for point_class in POINT_CLASSES:
        count = int((context["pointId"].astype(int) == point_class).sum())
        row[f"point_count_{point_class}"] = count
        row[f"point_ratio_{point_class}"] = float(count / max(context_len, 1))

    strike = context["strikeId"].astype(int)
    row["serve_count"] = int((strike == 1).sum())
    row["receive_count"] = int((strike == 2).sum())
    row["rally_stroke_count"] = int((strike == 4).sum())
    row["serve_ratio"] = float(row["serve_count"] / max(context_len, 1))
    row["receive_ratio"] = float(row["receive_count"] / max(context_len, 1))
    row["rally_stroke_ratio"] = float(row["rally_stroke_count"] / max(context_len, 1))


def _mode_int(values: np.ndarray, default: int = PAD_CAT) -> int:
    if len(values) == 0:
        return default
    counts = pd.Series(values.astype(int)).value_counts(sort=True)
    return int(counts.index[0])


def _is_categorical_feature_column(column: str) -> bool:
    direct_columns = {
        "match",
        "rally_id",
        "context_bucket",
        "target_stage",
        "next_strikeId",
        "score_state",
        "lead_state",
        "is_tied",
        "is_deuce_like",
        "sex",
        "numberGame",
        "next_player_id",
        "next_opponent_id",
        "player_pair_id",
        "player_pair_unordered_id",
    }
    if column in direct_columns:
        return True

    for role_name in ("next_player", "next_opponent"):
        if column.startswith(f"{role_name}_last_") and column[len(f"{role_name}_last_") :] in ROLE_HISTORY_COLS:
            return True

    if column.startswith("lag"):
        if column.endswith("_is_available") or column.endswith("_player_pair_id") or column.endswith("_transition"):
            return True
        for suffix in LAST_K_STROKE_COLS:
            if column.endswith(f"_{suffix}"):
                return True

    for agg_column in AGG_COLS:
        if column in {f"{agg_column}_first", f"{agg_column}_last", f"{agg_column}_mode"}:
            return True

    return False
