"""Train AutoGluon tabular models with cowork-style preprocessing and OOF scoring.

Examples:
    python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 8 --folds 5 --time-limit 600
    python src/train_automl.py --train-path data/train.csv --model-dir models/debug_automl --k 5 --folds 3 --time-limit 60
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from features import (
    ACTION_CLASSES,
    FORBIDDEN_FEATURE_COLUMNS,
    POINT_CLASSES,
    PREPROCESSING_SIGNATURE,
    TARGET_COLUMNS,
    align_feature_columns,
    apply_categorical_dtypes,
    apply_unknown_player_mapping,
    build_train_features,
    collect_known_players,
    get_categorical_feature_columns,
    get_model_feature_columns,
    make_group_folds,
)
from metrics import safe_auc, score_from_probabilities

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
    "point_zero": {
        "label": "target_point_is_zero",
        "problem_type": "binary",
        "eval_metric": "roc_auc",
    },
    "point_nonzero": {
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

OVERALL_SCORE_WEIGHTS = {"action": 0.4, "point": 0.4, "win": 0.2}


@dataclass
class PredictorArtifacts:
    predictor: Any
    model_name: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AutoGluon baselines with cowork-style prefix preprocessing and OOF scoring.",
        epilog=(
            "Examples:\n"
            "  python src/train_automl.py --train-path data/train.csv --model-dir models/automl --k 8 --folds 5 --time-limit 600\n"
            "  python src/train_automl.py --train-path data/train.csv --model-dir models/debug_automl --k 5 --folds 3 --time-limit 60"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-path", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--model-dir", default="models/automl", help="Directory to save AutoGluon models")
    parser.add_argument("--k", type=int, default=8, help="Number of lag strokes to use")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold folds for OOF evaluation")
    parser.add_argument("--time-limit", type=int, default=600, help="Per-model training time limit in seconds")
    parser.add_argument("--presets", default="medium_quality", help="AutoGluon presets")
    parser.add_argument(
        "--point-two-stage",
        type=_int_flag,
        default=1,
        choices=(0, 1),
        help="Use a binary point-zero model plus a nonzero multiclass point model",
    )
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


def ensure_feature_columns(feature_columns: Sequence[str]) -> None:
    overlap = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if overlap:
        raise ValueError(f"Forbidden feature columns detected: {sorted(overlap)}")


def get_positive_class_probability(proba_df: pd.DataFrame) -> np.ndarray:
    if 1 in proba_df.columns:
        return proba_df[1].to_numpy(dtype=np.float32)
    if "1" in proba_df.columns:
        return proba_df["1"].to_numpy(dtype=np.float32)
    raise ValueError(f"Could not find positive class 1 in probability columns: {list(proba_df.columns)}")


def normalize_class_label(value: object) -> int:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, str):
        return int(float(value))
    return int(value)


def align_multiclass_probabilities(proba_df: pd.DataFrame, classes: Sequence[int]) -> np.ndarray:
    aligned = np.zeros((len(proba_df), len(classes)), dtype=np.float32)
    for column_idx, column_name in enumerate(proba_df.columns):
        class_id = normalize_class_label(column_name)
        if class_id in classes:
            aligned[:, classes.index(class_id)] = proba_df.iloc[:, column_idx].to_numpy(dtype=np.float32)
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-8)


def build_label_frame(features: pd.DataFrame, labels: pd.Series, label_name: str) -> pd.DataFrame:
    frame = features.copy()
    frame[label_name] = labels.to_numpy()
    return frame


def fit_predictor(
    name: str,
    spec: Dict[str, str],
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    predictor_path: Path,
    time_limit: int,
    presets: str,
    valid_features: pd.DataFrame | None = None,
    valid_labels: pd.Series | None = None,
    leaderboard_path: Path | None = None,
) -> PredictorArtifacts:
    from autogluon.tabular import TabularPredictor

    label = spec["label"]
    train_frame = build_label_frame(train_features, train_labels, label)
    valid_frame = None
    if valid_features is not None and valid_labels is not None:
        valid_frame = build_label_frame(valid_features, valid_labels, label)
        if valid_frame.empty:
            raise ValueError(f"{name} validation frame is empty")

    predictor = TabularPredictor(
        label=label,
        path=str(predictor_path),
        problem_type=spec["problem_type"],
        eval_metric=spec["eval_metric"],
    )

    fit_kwargs: Dict[str, Any] = {
        "train_data": train_frame,
        "presets": presets,
        "time_limit": time_limit,
    }
    if valid_frame is not None:
        fit_kwargs["tuning_data"] = valid_frame
        if presets in {"good_quality", "high_quality", "best_quality"}:
            fit_kwargs["use_bag_holdout"] = True
            fit_kwargs["dynamic_stacking"] = False
            fit_kwargs["num_stack_levels"] = 1

    predictor.fit(**fit_kwargs)

    model_name = str(predictor.model_best)
    if valid_frame is not None and leaderboard_path is not None:
        leaderboard = predictor.leaderboard(valid_frame, silent=True)
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(leaderboard_path, index=False)
        scored_rows = leaderboard[leaderboard["score_val"].notna()]
        if not scored_rows.empty:
            model_name = str(scored_rows.iloc[0]["model"])

    return PredictorArtifacts(predictor=predictor, model_name=model_name, label=label)


def build_two_stage_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    point_feature_df = feature_df.copy()
    point_feature_df["target_point_is_zero"] = (point_feature_df["target_pointId"] == 0).astype(int)
    return point_feature_df


def fit_fold_predictors(
    feature_df: pd.DataFrame,
    train_index: pd.Index,
    valid_index: pd.Index,
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    model_dir: Path,
    fold_number: int,
    time_limit: int,
    presets: str,
    point_strategy: str,
) -> Dict[str, PredictorArtifacts]:
    fold_dir = model_dir / "_cv" / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    point_feature_df = build_two_stage_feature_df(feature_df)

    predictors: Dict[str, PredictorArtifacts] = {}
    predictors["action"] = fit_predictor(
        name="action",
        spec=MODEL_SPECS["action"],
        train_features=train_features,
        train_labels=feature_df.loc[train_index, MODEL_SPECS["action"]["label"]],
        valid_features=valid_features,
        valid_labels=feature_df.loc[valid_index, MODEL_SPECS["action"]["label"]],
        predictor_path=fold_dir / "action",
        leaderboard_path=fold_dir / "action_leaderboard.csv",
        time_limit=time_limit,
        presets=presets,
    )

    if point_strategy == "single_stage":
        predictors["point"] = fit_predictor(
            name="point",
            spec=MODEL_SPECS["point"],
            train_features=train_features,
            train_labels=feature_df.loc[train_index, MODEL_SPECS["point"]["label"]],
            valid_features=valid_features,
            valid_labels=feature_df.loc[valid_index, MODEL_SPECS["point"]["label"]],
            predictor_path=fold_dir / "point",
            leaderboard_path=fold_dir / "point_leaderboard.csv",
            time_limit=time_limit,
            presets=presets,
        )
    else:
        predictors["point_zero"] = fit_predictor(
            name="point_zero",
            spec=MODEL_SPECS["point_zero"],
            train_features=train_features,
            train_labels=point_feature_df.loc[train_index, MODEL_SPECS["point_zero"]["label"]],
            valid_features=valid_features,
            valid_labels=point_feature_df.loc[valid_index, MODEL_SPECS["point_zero"]["label"]],
            predictor_path=fold_dir / "point_zero",
            leaderboard_path=fold_dir / "point_zero_leaderboard.csv",
            time_limit=time_limit,
            presets=presets,
        )

        nonzero_mask = point_feature_df["target_pointId"] != 0
        nonzero_train_index = train_index.intersection(point_feature_df.index[nonzero_mask])
        nonzero_valid_index = valid_index.intersection(point_feature_df.index[nonzero_mask])
        if nonzero_train_index.empty or nonzero_valid_index.empty:
            raise ValueError("Two-stage point training needs nonzero point samples in both train and valid folds")

        predictors["point_nonzero"] = fit_predictor(
            name="point_nonzero",
            spec=MODEL_SPECS["point_nonzero"],
            train_features=train_features.loc[nonzero_train_index],
            train_labels=point_feature_df.loc[nonzero_train_index, MODEL_SPECS["point_nonzero"]["label"]],
            valid_features=valid_features.loc[nonzero_valid_index],
            valid_labels=point_feature_df.loc[nonzero_valid_index, MODEL_SPECS["point_nonzero"]["label"]],
            predictor_path=fold_dir / "point_nonzero",
            leaderboard_path=fold_dir / "point_nonzero_leaderboard.csv",
            time_limit=time_limit,
            presets=presets,
        )

    predictors["win"] = fit_predictor(
        name="win",
        spec=MODEL_SPECS["win"],
        train_features=train_features,
        train_labels=feature_df.loc[train_index, MODEL_SPECS["win"]["label"]],
        valid_features=valid_features,
        valid_labels=feature_df.loc[valid_index, MODEL_SPECS["win"]["label"]],
        predictor_path=fold_dir / "win",
        leaderboard_path=fold_dir / "win_leaderboard.csv",
        time_limit=time_limit,
        presets=presets,
    )

    return predictors


def fit_final_predictors(
    feature_df: pd.DataFrame,
    full_features: pd.DataFrame,
    model_dir: Path,
    time_limit: int,
    presets: str,
    point_strategy: str,
) -> Dict[str, PredictorArtifacts]:
    point_feature_df = build_two_stage_feature_df(feature_df)

    predictors: Dict[str, PredictorArtifacts] = {}
    predictors["action"] = fit_predictor(
        name="action",
        spec=MODEL_SPECS["action"],
        train_features=full_features,
        train_labels=feature_df[MODEL_SPECS["action"]["label"]],
        predictor_path=model_dir / "action",
        time_limit=time_limit,
        presets=presets,
    )

    if point_strategy == "single_stage":
        predictors["point"] = fit_predictor(
            name="point",
            spec=MODEL_SPECS["point"],
            train_features=full_features,
            train_labels=feature_df[MODEL_SPECS["point"]["label"]],
            predictor_path=model_dir / "point",
            time_limit=time_limit,
            presets=presets,
        )
    else:
        predictors["point_zero"] = fit_predictor(
            name="point_zero",
            spec=MODEL_SPECS["point_zero"],
            train_features=full_features,
            train_labels=point_feature_df[MODEL_SPECS["point_zero"]["label"]],
            predictor_path=model_dir / "point_zero",
            time_limit=time_limit,
            presets=presets,
        )
        nonzero_mask = point_feature_df["target_pointId"] != 0
        predictors["point_nonzero"] = fit_predictor(
            name="point_nonzero",
            spec=MODEL_SPECS["point_nonzero"],
            train_features=full_features.loc[nonzero_mask],
            train_labels=point_feature_df.loc[nonzero_mask, MODEL_SPECS["point_nonzero"]["label"]],
            predictor_path=model_dir / "point_nonzero",
            time_limit=time_limit,
            presets=presets,
        )

    predictors["win"] = fit_predictor(
        name="win",
        spec=MODEL_SPECS["win"],
        train_features=full_features,
        train_labels=feature_df[MODEL_SPECS["win"]["label"]],
        predictor_path=model_dir / "win",
        time_limit=time_limit,
        presets=presets,
    )

    return predictors


def predict_point_probabilities(
    valid_features: pd.DataFrame,
    predictors: Dict[str, PredictorArtifacts],
    point_strategy: str,
) -> tuple[np.ndarray, dict[str, float | None]]:
    if point_strategy == "single_stage":
        point_proba_df = predictors["point"].predictor.predict_proba(
            valid_features,
            model=predictors["point"].model_name,
        )
        point_prob = align_multiclass_probabilities(point_proba_df, POINT_CLASSES)
        return point_prob, {"point_zero_roc_auc": None}

    point_zero_proba_df = predictors["point_zero"].predictor.predict_proba(
        valid_features,
        model=predictors["point_zero"].model_name,
    )
    point_zero_proba = get_positive_class_probability(point_zero_proba_df)

    point_nonzero_proba_df = predictors["point_nonzero"].predictor.predict_proba(
        valid_features,
        model=predictors["point_nonzero"].model_name,
    )
    point_nonzero_prob = align_multiclass_probabilities(point_nonzero_proba_df, POINT_CLASSES)
    combined = point_nonzero_prob * (1.0 - point_zero_proba[:, None])
    combined[:, 0] = point_zero_proba
    row_sum = combined.sum(axis=1, keepdims=True)
    combined = combined / np.maximum(row_sum, 1e-8)
    return combined, {"point_zero_roc_auc": None}


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    point_strategy = "two_stage" if args.point_two_stage else "single_stage"
    include_match_feature = bool(args.include_match_feature)
    include_rally_id_feature = bool(args.include_rally_id_feature)

    train_df = pd.read_csv(args.train_path)
    feature_df = build_train_features(train_df, k=args.k, cast_categoricals=False)
    feature_columns = get_model_feature_columns(
        feature_df,
        include_match_feature=include_match_feature,
        include_rally_id_feature=include_rally_id_feature,
    )
    ensure_feature_columns(feature_columns)
    categorical_feature_columns = get_categorical_feature_columns(feature_columns)

    if not set(TARGET_COLUMNS).issubset(feature_df.columns):
        raise ValueError(f"Missing required target columns in feature DataFrame: {TARGET_COLUMNS}")

    folds, fold_info = make_group_folds(feature_df, n_splits=args.folds)
    y_action = feature_df["target_actionId"].to_numpy(dtype=np.int64)
    y_point = feature_df["target_pointId"].to_numpy(dtype=np.int64)
    y_server = feature_df["target_serverGetPoint"].to_numpy(dtype=np.float32)

    oof_action = np.zeros((len(feature_df), len(ACTION_CLASSES)), dtype=np.float32)
    oof_point = np.zeros((len(feature_df), len(POINT_CLASSES)), dtype=np.float32)
    oof_server = np.zeros(len(feature_df), dtype=np.float32)
    fold_scores: List[Dict[str, Any]] = []

    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        known_players = sorted(collect_known_players(feature_df.loc[train_index, feature_columns], args.k))
        train_features = apply_unknown_player_mapping(feature_df.loc[train_index, feature_columns], known_players, args.k)
        valid_features = apply_unknown_player_mapping(feature_df.loc[valid_index, feature_columns], known_players, args.k)
        train_features = align_feature_columns(train_features, feature_columns, categorical_feature_columns)
        valid_features = align_feature_columns(valid_features, feature_columns, categorical_feature_columns)

        predictors = fit_fold_predictors(
            feature_df=feature_df,
            train_index=train_index,
            valid_index=valid_index,
            train_features=train_features,
            valid_features=valid_features,
            model_dir=model_dir,
            fold_number=fold_number,
            time_limit=args.time_limit,
            presets=args.presets,
            point_strategy=point_strategy,
        )

        action_proba_df = predictors["action"].predictor.predict_proba(
            valid_features,
            model=predictors["action"].model_name,
        )
        action_prob = align_multiclass_probabilities(action_proba_df, ACTION_CLASSES)
        point_prob, point_details = predict_point_probabilities(valid_features, predictors, point_strategy)
        win_proba_df = predictors["win"].predictor.predict_proba(
            valid_features,
            model=predictors["win"].model_name,
        )
        win_prob = get_positive_class_probability(win_proba_df)

        oof_action[valid_index] = action_prob
        oof_point[valid_index] = point_prob
        oof_server[valid_index] = win_prob

        fold_score = score_from_probabilities(
            y_action[valid_index],
            action_prob,
            y_point[valid_index],
            point_prob,
            y_server[valid_index],
            win_prob,
        )
        point_zero_auc = None
        if point_strategy == "two_stage":
            point_zero_true = (y_point[valid_index] == 0).astype(np.int64)
            point_zero_auc = safe_auc(point_zero_true, point_prob[:, 0])

        fold_record = {
            "fold": fold_number,
            "n_train_rows": int(len(train_index)),
            "n_valid_rows": int(len(valid_index)),
            "known_player_count": int(len(known_players)),
            "score": fold_score,
            "point_zero_roc_auc": point_zero_auc if point_strategy == "two_stage" else point_details["point_zero_roc_auc"],
            "validation_models": {name: artifacts.model_name for name, artifacts in predictors.items()},
        }
        fold_scores.append(fold_record)
        print(f"Fold {fold_number}/{args.folds} score: {fold_score}")

    oof_score = score_from_probabilities(y_action, oof_action, y_point, oof_point, y_server, oof_server)
    print(f"OOF score: {oof_score}")

    full_known_players = sorted(collect_known_players(feature_df.loc[:, feature_columns], args.k))
    full_features = apply_unknown_player_mapping(feature_df.loc[:, feature_columns], full_known_players, args.k)
    full_features = align_feature_columns(full_features, feature_columns, categorical_feature_columns)
    fit_final_predictors(
        feature_df=feature_df,
        full_features=full_features,
        model_dir=model_dir,
        time_limit=args.time_limit,
        presets=args.presets,
        point_strategy=point_strategy,
    )

    metadata = {
        "k": args.k,
        "folds": args.folds,
        "train_path": args.train_path,
        "point_strategy": point_strategy,
        "preprocessing_signature": PREPROCESSING_SIGNATURE,
        "action_classes": ACTION_CLASSES,
        "point_classes": POINT_CLASSES,
        "feature_columns": feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "feature_selection": {
            "include_match_feature": include_match_feature,
            "include_rally_id_feature": include_rally_id_feature,
        },
        "known_players": full_known_players,
        "group_folds": {
            "group_column": fold_info.group_column,
            "n_splits": fold_info.n_splits,
            "n_rows": fold_info.n_rows,
            "n_groups": fold_info.n_groups,
        },
        "score_weights": OVERALL_SCORE_WEIGHTS,
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    validation_summary = {
        "oof": oof_score,
        "folds": fold_scores,
        "point_strategy": point_strategy,
        "group_column": fold_info.group_column,
        "preprocessing_signature": PREPROCESSING_SIGNATURE,
    }
    (model_dir / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    np.savez_compressed(
        model_dir / "oof_predictions.npz",
        oof_action=oof_action,
        oof_point=oof_point,
        oof_server=oof_server,
        y_action=y_action,
        y_point=y_point,
        y_server=y_server,
        oof_rally_uid=feature_df["rally_uid"].to_numpy(dtype=np.int64),
        oof_context_len=feature_df["context_len"].to_numpy(dtype=np.int64),
        oof_rally_len=feature_df["rally_len"].to_numpy(dtype=np.int64),
        oof_target_strike_number=feature_df["target_strikeNumber"].to_numpy(dtype=np.int64),
        oof_is_last_target=feature_df["is_last_target"].to_numpy(dtype=np.int8),
    )

    print(f"Saved models, metadata, and OOF artifacts under: {model_dir}")


if __name__ == "__main__":
    main()
