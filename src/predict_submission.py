"""Generate competition submission CSV from trained AutoGluon predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from features import (
    ACTION_CLASSES,
    POINT_CLASSES,
    PREPROCESSING_SIGNATURE,
    align_feature_columns,
    apply_unknown_player_mapping,
    build_test_features,
)

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict submission rows from trained AutoGluon models.",
        epilog=(
            "Example:\n"
            "  python src/predict_submission.py --test-path data/test_new.csv --model-dir models/automl --k 8 --out-path submission.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--test-path", default="data/test_new.csv", help="Path to test_new.csv")
    parser.add_argument("--model-dir", default="models/automl", help="Directory containing trained predictors")
    parser.add_argument("--k", type=int, default=8, help="Number of lag strokes to use")
    parser.add_argument("--out-path", default="submission.csv", help="Output CSV path")
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing training metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def get_positive_class_probability(proba_df: pd.DataFrame) -> pd.Series:
    if 1 in proba_df.columns:
        return proba_df[1]
    if "1" in proba_df.columns:
        return proba_df["1"]
    raise ValueError(f"Could not find positive class 1 in probability columns: {list(proba_df.columns)}")


def normalize_class_label(value: object) -> int:
    if isinstance(value, str):
        return int(float(value))
    return int(value)


def align_multiclass_probabilities(proba_df: pd.DataFrame, classes: List[int]) -> pd.DataFrame:
    aligned = pd.DataFrame(0.0, index=proba_df.index, columns=classes)
    for column_name in proba_df.columns:
        class_id = normalize_class_label(column_name)
        if class_id in aligned.columns:
            aligned[class_id] = proba_df[column_name].astype(float)
    row_sum = aligned.sum(axis=1)
    row_sum = row_sum.where(row_sum > 0, 1.0)
    return aligned.div(row_sum, axis=0)


def assert_integer_like(values: Iterable[object], column_name: str) -> pd.Series:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="raise")
    if not (numeric == numeric.astype(int)).all():
        raise ValueError(f"{column_name} contains non-integer predictions")
    return numeric.astype(int)


def validate_submission(submission_df: pd.DataFrame, expected_rally_order: List[object], sample_path: Path) -> None:
    if submission_df.columns.tolist() != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
    if len(submission_df) != len(expected_rally_order):
        raise ValueError("Submission row count does not match the number of unique test rallies")
    if submission_df["rally_uid"].tolist() != expected_rally_order:
        raise ValueError("Submission rally_uid order does not match test_new.csv first-appearance order")
    if sample_path.exists():
        sample_columns = pd.read_csv(sample_path, nrows=0).columns.tolist()
        if sample_columns != SUBMISSION_COLUMNS:
            raise ValueError(f"Hardcoded submission columns differ from sample_submission.csv: {sample_columns}")
    if not pd.api.types.is_float_dtype(submission_df["serverGetPoint"]):
        raise ValueError("Submission serverGetPoint must be float probabilities")


def predict_point_ids(
    inference_frame: pd.DataFrame,
    model_dir: Path,
    point_strategy: str,
) -> pd.Series:
    from autogluon.tabular import TabularPredictor

    if point_strategy == "single_stage":
        point_predictor = TabularPredictor.load(str(model_dir / "point"))
        point_prob = align_multiclass_probabilities(point_predictor.predict_proba(inference_frame), POINT_CLASSES)
        return assert_integer_like(point_prob.idxmax(axis=1), "pointId")

    point_zero_predictor = TabularPredictor.load(str(model_dir / "point_zero"))
    point_nonzero_predictor = TabularPredictor.load(str(model_dir / "point_nonzero"))

    point_zero_proba = get_positive_class_probability(point_zero_predictor.predict_proba(inference_frame)).astype(float)
    point_nonzero_prob = align_multiclass_probabilities(
        point_nonzero_predictor.predict_proba(inference_frame),
        POINT_CLASSES,
    )
    combined = point_nonzero_prob.mul((1.0 - point_zero_proba).to_numpy(), axis=0)
    combined[0] = point_zero_proba.to_numpy()
    row_sum = combined.sum(axis=1).where(lambda values: values > 0, 1.0)
    combined = combined.div(row_sum, axis=0)
    return assert_integer_like(combined.idxmax(axis=1), "pointId")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metadata = load_metadata(model_dir)
    trained_k = metadata.get("k")
    if trained_k != args.k:
        raise ValueError(f"Provided k={args.k} does not match trained metadata k={trained_k}")

    preprocessing_signature = metadata.get("preprocessing_signature")
    if preprocessing_signature != PREPROCESSING_SIGNATURE:
        raise ValueError(
            f"Unsupported preprocessing signature {preprocessing_signature!r}; expected {PREPROCESSING_SIGNATURE!r}"
        )

    point_strategy = metadata.get("point_strategy", "single_stage")
    feature_columns = metadata["feature_columns"]
    categorical_feature_columns = metadata.get("categorical_feature_columns") or []
    known_players = metadata.get("known_players") or []

    test_df = pd.read_csv(args.test_path)
    rally_order = test_df["rally_uid"].drop_duplicates().tolist()
    feature_df = build_test_features(test_df, k=args.k, cast_categoricals=False)
    feature_df = apply_unknown_player_mapping(feature_df, known_players, args.k)
    inference_frame = align_feature_columns(feature_df, feature_columns, categorical_feature_columns)

    from autogluon.tabular import TabularPredictor

    action_predictor = TabularPredictor.load(str(model_dir / "action"))
    win_predictor = TabularPredictor.load(str(model_dir / "win"))

    action_prob = align_multiclass_probabilities(action_predictor.predict_proba(inference_frame), ACTION_CLASSES)
    action_pred = assert_integer_like(action_prob.idxmax(axis=1), "actionId")
    point_pred = predict_point_ids(inference_frame, model_dir=model_dir, point_strategy=point_strategy)
    win_proba = get_positive_class_probability(win_predictor.predict_proba(inference_frame)).astype(float)

    submission_df = pd.DataFrame(
        {
            "rally_uid": feature_df["rally_uid"].tolist(),
            "actionId": action_pred.tolist(),
            "pointId": point_pred.tolist(),
            "serverGetPoint": win_proba.tolist(),
        }
    )

    sample_path = Path(args.test_path).with_name("sample_submission.csv")
    validate_submission(submission_df, expected_rally_order=rally_order, sample_path=sample_path)

    out_path = Path(args.out_path)
    submission_df.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
