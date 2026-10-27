"""Metrics from the brief, Sec 6. Pure numpy/sklearn -- real and runnable without torch."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


def multilabel_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Average precision per tag, mean over tags with at least one positive.
    ap_scores = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        ap_scores.append(average_precision_score(y_true[:, k], y_prob[:, k]))
    mean_auc_pr = float(np.mean(ap_scores)) if ap_scores else 0.0

    precision, recall, f1_per_tag, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "mean_auc_pr": mean_auc_pr,
        "per_tag": {"precision": precision.tolist(), "recall": recall.tolist(), "f1": f1_per_tag.tolist(), "support": support.tolist()},
    }


def emotion_report(v_true, v_pred, a_true, a_pred) -> dict:
    def mae(y, yhat):
        return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))

    def r2(y, yhat):
        y, yhat = np.asarray(y), np.asarray(yhat)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "valence_mae": mae(v_true, v_pred),
        "valence_r2": r2(v_true, v_pred),
        "arousal_mae": mae(a_true, a_pred),
        "arousal_r2": r2(a_true, a_pred),
    }
