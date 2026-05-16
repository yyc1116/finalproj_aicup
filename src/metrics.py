"""Competition metrics shared by training and evaluation scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from features import ACTION_CLASSES, POINT_CLASSES


def macro_f1_action(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def macro_f1_point(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def competition_score(
    y_action: np.ndarray,
    pred_action: np.ndarray,
    y_point: np.ndarray,
    pred_point: np.ndarray,
    y_server: np.ndarray,
    prob_server: np.ndarray,
) -> dict[str, float]:
    action_f1 = macro_f1_action(y_action, pred_action)
    point_f1 = macro_f1_point(y_point, pred_point)
    server_auc = safe_auc(y_server, prob_server)
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "action_f1": action_f1,
        "point_f1": point_f1,
        "server_auc": server_auc,
        "overall": float(overall),
    }


def score_from_probabilities(
    y_action: np.ndarray,
    action_prob: np.ndarray,
    y_point: np.ndarray,
    point_prob: np.ndarray,
    y_server: np.ndarray,
    server_prob: np.ndarray,
) -> dict[str, float]:
    return competition_score(
        y_action=y_action,
        pred_action=action_prob.argmax(axis=1),
        y_point=y_point,
        pred_point=point_prob.argmax(axis=1),
        y_server=y_server,
        prob_server=server_prob,
    )
