from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ..config import LOSOResult, ModelMeta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Type aliases
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]

NON_FEATURE_COLS: set[str] = {"session", "label", "n_samples"}
POS_LABEL: str = "drowsy"

def build_models() -> dict[str, Pipeline]:
    """Return name -> sklearn Pipeline for each candidate model."""
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=0)),
        ]),
    }


def loso_predict(
    model: Pipeline,
    X: FloatArray,
    y: IntArray,
    groups: npt.NDArray[np.str_],
) -> LOSOResult:
    """Leave-one-session-out CV returning pooled out-of-fold predictions."""
    logo: LeaveOneGroupOut = LeaveOneGroupOut()
    yt: list[IntArray] = []
    yp: list[IntArray] = []
    pp: list[FloatArray] = []
    gg: list[npt.NDArray[np.str_]] = []

    for tr, te in logo.split(X, y, groups):
        model.fit(X[tr], y[tr])
        yp.append(model.predict(X[te]))
        classes: list[int] = list(model.classes_)
        prob: FloatArray = model.predict_proba(X[te])[:, classes.index(1)]
        pp.append(prob)
        yt.append(y[te])
        gg.append(groups[te])

    name: str = [k for k, v in build_models().items()
                 if type(v.named_steps["clf"]) is type(model.named_steps["clf"])][0]

    return LOSOResult(
        model_name=name,
        y_true=np.concatenate(yt).tolist(),
        y_pred=np.concatenate(yp).tolist(),
        y_prob=np.concatenate(pp).tolist(),
        groups=np.concatenate(gg).tolist(),
    )

LABEL_NAMES: dict[int, str] = {0: "alert", 1: "drowsy"}


def report(result: LOSOResult) -> None:
    """Print classification metrics for one model."""
    yt: IntArray = np.array(result.y_true)
    yp: IntArray = np.array(result.y_pred)
    yprob: FloatArray = np.array(result.y_prob)
    groups: npt.NDArray[np.str_] = np.array(result.groups)

    print(f"\n{'='*64}\n{result.model_name}\n{'='*64}")
    print("pooled out-of-fold metrics (positive class = drowsy):")
    print(f"  accuracy           {accuracy_score(yt, yp):.3f}")
    print(f"  balanced accuracy  {balanced_accuracy_score(yt, yp):.3f}")
    print(f"  precision          {precision_score(yt, yp, zero_division=0):.3f}")
    print(f"  recall             {recall_score(yt, yp, zero_division=0):.3f}")
    print(f"  f1                 {f1_score(yt, yp, zero_division=0):.3f}")
    if len(np.unique(yt)) == 2:
        print(f"  ROC-AUC            {roc_auc_score(yt, yprob):.3f}")
        print(f"  PR-AUC (avg prec)  {average_precision_score(yt, yprob):.3f}")

    cm: IntArray = confusion_matrix(yt, yp, labels=[0, 1])
    print(f"\n  confusion matrix [rows=true, cols=pred]  (0=alert, 1=drowsy)")
    print(f"           pred_alert  pred_drowsy")
    print(f"  alert      {cm[0,0]:>6d}      {cm[0,1]:>6d}")
    print(f"  drowsy     {cm[1,0]:>6d}      {cm[1,1]:>6d}")

    print("\n  per-session majority vote:")
    correct: int = 0
    for s in np.unique(groups):
        mask: npt.NDArray[np.bool_] = groups == s
        true_lab: int = int(round(yt[mask].mean()))
        vote: int = int(round(yp[mask].mean()))
        ok: str = "OK " if vote == true_lab else "XX "
        correct += int(vote == true_lab)
        print(f"    {ok}{s:42s} true={LABEL_NAMES[true_lab]:6s} "
              f"vote={LABEL_NAMES[vote]:6s} "
              f"({int((yp[mask]==1).sum())}/{mask.sum()} windows -> drowsy)")
    print(f"  sessions correct: {correct}/{len(np.unique(groups))}")


def _generate_plots(
    results: dict[str, LOSOResult],
    feat_cols: list[str],
    coefs: FloatArray,
    importances: FloatArray,
    plot_dir: str = "plots",
) -> None:
    """Generate all 7 training evaluation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating training plots -> {plot_dir}/")

    ALERT_COLOR: str = "#2196F3"
    DROWSY_COLOR: str = "#F44336"

    # Helper to get numpy arrays from a result
    def _arrays(name: str) -> tuple[IntArray, IntArray, FloatArray, npt.NDArray[np.str_]]:
        r: LOSOResult = results[name]
        return (np.array(r.y_true), np.array(r.y_pred),
                np.array(r.y_prob), np.array(r.groups))

    # 1. ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, color, ls in [("logistic_regression", "#2196F3", "-"),
                             ("random_forest", "#FF9800", "--")]:
        yt, _, yprob, _ = _arrays(name)
        auc: float = roc_auc_score(yt, yprob)
        fpr: FloatArray
        tpr: FloatArray
        fpr, tpr, _ = roc_curve(yt, yprob)
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (leave-one-session-out CV)", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "08_roc_curve.png"), dpi=150)
    plt.close(fig)
    print("  08_roc_curve.png")

    # 2. Precision-Recall curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, color, ls in [("logistic_regression", "#2196F3", "-"),
                             ("random_forest", "#FF9800", "--")]:
        yt, _, yprob, _ = _arrays(name)
        ap: float = average_precision_score(yt, yprob)
        prec: FloatArray
        rec: FloatArray
        prec, rec, _ = precision_recall_curve(yt, yprob)
        ax.plot(rec, prec, color=color, linestyle=ls, lw=2,
                label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (LOSO CV)", fontsize=13)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "09_precision_recall_curve.png"), dpi=150)
    plt.close(fig)
    print("  09_precision_recall_curve.png")

    # 3. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ["logistic_regression", "random_forest"]):
        yt, yp, _, _ = _arrays(name)
        cm: IntArray = confusion_matrix(yt, yp, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["alert", "drowsy"])
        ax.set_yticklabels(["alert", "drowsy"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc: float = accuracy_score(yt, yp)
        ax.set_title(f"{name}\nacc={acc:.3f}", fontsize=11)
    fig.suptitle("Confusion matrices (LOSO CV)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "10_confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  10_confusion_matrices.png")

    # 4. LR coefficients
    coef_order: IntArray = np.argsort(coefs)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors: list[str] = [
        DROWSY_COLOR if c > 0 else ALERT_COLOR for c in coefs[coef_order]]
    ax.barh(range(len(feat_cols)), coefs[coef_order], color=colors)
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels([feat_cols[i] for i in coef_order], fontsize=9)
    ax.set_xlabel("standardised coefficient")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Logistic regression coefficients\n"
                 "(red = more drowsy, blue = more alert)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "11_lr_coefficients.png"), dpi=150)
    plt.close(fig)
    print("  11_lr_coefficients.png")

    # 5. RF importances
    imp_order: IntArray = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(feat_cols)), importances[imp_order], color="#FF9800")
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels([feat_cols[i] for i in imp_order], fontsize=9)
    ax.set_xlabel("importance (Gini)")
    ax.set_title("Random forest feature importances", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "12_rf_importances.png"), dpi=150)
    plt.close(fig)
    print("  12_rf_importances.png")

    # 6. Per-session probability timeline
    r: LOSOResult = results["logistic_regression"]
    groups_arr: npt.NDArray[np.str_] = np.array(r.groups)
    unique_sess: list[str] = list(dict.fromkeys(r.groups))
    fig, axes = plt.subplots(
        len(unique_sess), 1, figsize=(12, 2.5 * len(unique_sess)),
        sharex=False, squeeze=False)
    for i, sess in enumerate(unique_sess):
        mask: npt.NDArray[np.bool_] = groups_arr == sess
        probs: FloatArray = np.array(r.y_prob)[mask]
        true_lab: int = int(round(np.array(r.y_true)[mask].mean()))
        lab_str: str = LABEL_NAMES[true_lab]
        color: str = DROWSY_COLOR if true_lab == 1 else ALERT_COLOR
        x: IntArray = np.arange(len(probs))
        axes[i, 0].bar(x, probs, color=color, alpha=0.7, width=1.0)
        axes[i, 0].axhline(0.5, color="black", ls="--", lw=0.8)
        axes[i, 0].fill_between(x, 0.5, 1.0, alpha=0.05, color=DROWSY_COLOR)
        axes[i, 0].fill_between(x, 0.0, 0.5, alpha=0.05, color=ALERT_COLOR)
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].set_ylabel("P(drowsy)")
        vote_str: str = "drowsy" if probs.mean() > 0.5 else "alert"
        axes[i, 0].set_title(
            f"{sess[:35]}  (true={lab_str}, vote={vote_str})",
            fontsize=10, color=color)
        axes[i, 0].grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xlabel("window index")
    fig.suptitle("Per-session drowsiness probability (LR, LOSO)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "13_session_probability_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  13_session_probability_timeline.png")

    # 7. Metrics comparison
    metric_names: list[str] = [
        "accuracy", "balanced_acc", "precision", "recall", "f1",
        "ROC-AUC", "PR-AUC"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos: FloatArray = np.arange(len(metric_names))
    width: float = 0.35
    for idx, (name, color) in enumerate([
        ("logistic_regression", "#2196F3"),
        ("random_forest", "#FF9800"),
    ]):
        yt, yp, yprob, _ = _arrays(name)
        vals: list[float] = [
            accuracy_score(yt, yp),
            balanced_accuracy_score(yt, yp),
            precision_score(yt, yp, zero_division=0),
            recall_score(yt, yp, zero_division=0),
            f1_score(yt, yp, zero_division=0),
            roc_auc_score(yt, yprob),
            average_precision_score(yt, yprob),
        ]
        bars = ax.bar(x_pos + idx * width, vals, width, label=name,
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score")
    ax.set_title("Model comparison (LOSO CV)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "14_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("  14_model_comparison.png")
    print(f"\nAll training plots saved to {plot_dir}/")

def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features.csv")
    args: argparse.Namespace = ap.parse_args()

    df: pd.DataFrame = pd.read_csv(args.features)
    feat_cols: list[str] = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X: FloatArray = df[feat_cols].to_numpy(dtype=float)
    groups: npt.NDArray[np.str_] = df["session"].to_numpy()
    y: IntArray = (df["label"].to_numpy() == POS_LABEL).astype(int)

    n_sessions: int = len(np.unique(groups))
    n_classes: int = len(np.unique(y))
    print(f"features: {feat_cols}")
    print(f"windows: {len(df)}   sessions: {n_sessions}   "
          f"class counts: {df['label'].value_counts().to_dict()}")

    if n_sessions < 2 or n_classes < 2:
        print("\nNeed >=2 sessions AND both classes. Re-run extract_features.")
        return

    # LOSO CV
    results: dict[str, LOSOResult] = {}
    for name, model in build_models().items():
        result: LOSOResult = loso_predict(model, X, y, groups)
        report(result)
        results[name] = result

    # Interpretability (fit on all data)
    print(f"\n{'='*64}\nfeature signal (fit on all data)\n{'='*64}")
    lr: Pipeline = build_models()["logistic_regression"].fit(X, y)
    coefs: FloatArray = lr.named_steps["clf"].coef_[0]
    order: IntArray = np.argsort(-np.abs(coefs))
    print("logistic-regression coefficients (standardised; + => more drowsy):")
    for i in order:
        print(f"  {feat_cols[i]:22s} {coefs[i]:+.3f}")

    rf: Pipeline = build_models()["random_forest"].fit(X, y)
    imp: FloatArray = rf.named_steps["clf"].feature_importances_
    order = np.argsort(-imp)
    print("\nrandom-forest feature importances:")
    for i in order:
        print(f"  {feat_cols[i]:22s} {imp[i]:.3f}")

    # Save model
    model_path: str = "drowsy_model.pkl"
    meta_path: str = "drowsy_model_meta.json"

    lr_final: Pipeline = build_models()["logistic_regression"].fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(lr_final, f)

    ext_meta_path: str = args.features.replace(".csv", "_meta.json")
    ext_meta: dict[str, Any] = {}
    try:
        with open(ext_meta_path) as f:
            ext_meta = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {ext_meta_path} not found; re-run extract_features.")

    meta: ModelMeta = ModelMeta(
        feature_cols=feat_cols,
        classes=[int(c) for c in lr_final.classes_],
        alpha=ext_meta.get("alpha"),
        fs=int(ext_meta.get("fs", 20)),
        window_s=float(ext_meta.get("window_s", 15.0)),
    )
    with open(meta_path, "w") as f:
        f.write(meta.model_dump_json(indent=2))

    print(f"\nSaved model  -> {model_path}")
    print(f"Saved meta   -> {meta_path}")

    # Plots
    _generate_plots(results, feat_cols, coefs, imp)


if __name__ == "__main__":
    main()
