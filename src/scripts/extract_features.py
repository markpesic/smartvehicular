from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal

from ..config import CONFIG, ExtractionMeta
from ..feature_extraction import (
    extract_features as compute_window_features,
    _detrend,
    _steering_prediction_errors,
)

# Type aliases
FloatArray = npt.NDArray[np.floating]

STEER_COL: str = "steer_cmd"
MIN_CLEAN_FRAC: float = CONFIG.features.reversal_gap 
_MIN_CLEAN_FRAC: float = 0.7

def _clean_row_mask(df: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Only junctions are filtered — speed drops, reverse, curvature spikes
    are kept because they are symptoms of drowsy overcorrection"""
    return (df["on_junction"].astype(int) == 0).to_numpy()


def _raw_segments(
    n_rows: int,
    frames: npt.NDArray[np.int64],
) -> list[tuple[int, int]]:
    """Index ranges that are contiguous in ``frame`` (handles logging gaps)"""
    breaks: FloatArray = np.where(np.diff(frames) != 1)[0]
    starts: FloatArray = np.concatenate(([0], breaks + 1))
    ends: FloatArray = np.concatenate((breaks + 1, [n_rows]))
    return list(zip(starts.tolist(), ends.tolist()))


def extract_session(
    path: str,
    win_s: float,
    overlap: float,
) -> tuple[str, str, int, list[FloatArray]]:
    """Window one session CSV and return accepted steering windows"""
    df: pd.DataFrame = pd.read_csv(path).sort_values("frame").reset_index(drop=True)
    label: str = str(df["session_type"].iloc[0])
    session: str = os.path.splitext(os.path.basename(path))[0]
    fs: int = round(1.0 / float(df["dt"].iloc[0]))

    steer: FloatArray = df[STEER_COL].astype(float).to_numpy()
    not_junction: FloatArray = _clean_row_mask(df).astype(float)
    frames: npt.NDArray[np.int64] = df["frame"].astype(np.int64).to_numpy()

    win: int = int(round(win_s * fs))
    step: int = max(1, int(round(win * (1 - overlap))))

    windows: list[FloatArray] = []
    for a, b in _raw_segments(len(df), frames):
        last_start: int = b - win
        for start in range(a, last_start + 1, step):
            sl: slice = slice(start, start + win)
            if not_junction[sl].mean() >= _MIN_CLEAN_FRAC:
                windows.append(steer[sl])
    return session, label, fs, windows


def _gather_prediction_errors(
    sessions: list[tuple[str, str, int, list[FloatArray]]],
    only_label: str | None = None,
) -> FloatArray:
    """Concatenate steering prediction errors across sessions."""
    errs: list[FloatArray] = []
    for _, label, _, windows in sessions:
        if only_label is not None and label != only_label:
            continue
        for w in windows:
            e: FloatArray = _steering_prediction_errors(
                _detrend(np.asarray(w, dtype=float)))
            if len(e):
                errs.append(e)
    return np.concatenate(errs) if errs else np.array([], dtype=float)

def _generate_plots(
    paths: list[str],
    sessions: list[tuple[str, str, int, list[FloatArray]]],
    out_df: pd.DataFrame,
    feat_cols: list[str],
    fs_global: int,
    plot_dir: str = "plots",
) -> None:
    """Generate all 7 data-exploration plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots -> {plot_dir}/")

    ALERT_COLOR: str = "#2196F3"
    DROWSY_COLOR: str = "#F44336"
    label_colors: dict[str, str] = {"alert": ALERT_COLOR, "drowsy": DROWSY_COLOR}
    lf_band: tuple[float, float] = CONFIG.features.lf_band

    # 1. Raw steering traces
    fig, axes = plt.subplots(
        len(paths), 1, figsize=(14, 3 * len(paths)),
        sharex=False, squeeze=False)
    for i, p in enumerate(paths):
        df_raw: pd.DataFrame = pd.read_csv(p)
        lab: str = str(df_raw["session_type"].iloc[0])
        t: pd.Series = df_raw["sim_time"].astype(float) - df_raw["sim_time"].astype(float).iloc[0]
        axes[i, 0].plot(t, df_raw["steer_cmd"].astype(float),
                        linewidth=0.3, color=label_colors.get(lab, "gray"))
        axes[i, 0].set_ylabel("steering")
        name: str = os.path.basename(p).replace(".csv", "")
        axes[i, 0].set_title(f"{name}  ({lab})", fontsize=10,
                             color=label_colors.get(lab, "black"))
        axes[i, 0].set_ylim(-1.1, 1.1)
        axes[i, 0].axhline(0, color="gray", lw=0.5)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Raw steering traces per session", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "01_raw_steering_traces.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  01_raw_steering_traces.png")

    # 2. Example windows
    alert_wins: list[tuple[str, FloatArray]] = [
        (s, w) for s, l, _, ws in sessions if l == "alert"
        for w in ws for s in [s]]
    drowsy_wins: list[tuple[str, FloatArray]] = [
        (s, w) for s, l, _, ws in sessions if l == "drowsy"
        for w in ws for s in [s]]
    n_ex: int = min(3, len(alert_wins), len(drowsy_wins))
    if n_ex > 0:
        rng: np.random.Generator = np.random.default_rng(42)
        a_idx: FloatArray = rng.choice(len(alert_wins), n_ex, replace=False)
        d_idx: FloatArray = rng.choice(len(drowsy_wins), n_ex, replace=False)
        fig, axes = plt.subplots(n_ex, 2, figsize=(12, 2.5 * n_ex), squeeze=False)
        for row in range(n_ex):
            t_ax: FloatArray = np.arange(len(alert_wins[a_idx[row]][1])) / fs_global
            axes[row, 0].plot(t_ax, alert_wins[a_idx[row]][1],
                              color=ALERT_COLOR, lw=0.6)
            axes[row, 0].set_ylim(-1.1, 1.1)
            axes[row, 0].axhline(0, color="gray", lw=0.3)
            if row == 0:
                axes[row, 0].set_title("ALERT windows", color=ALERT_COLOR,
                                       fontweight="bold")
            t_ax = np.arange(len(drowsy_wins[d_idx[row]][1])) / fs_global
            axes[row, 1].plot(t_ax, drowsy_wins[d_idx[row]][1],
                              color=DROWSY_COLOR, lw=0.6)
            axes[row, 1].set_ylim(-1.1, 1.1)
            axes[row, 1].axhline(0, color="gray", lw=0.3)
            if row == 0:
                axes[row, 1].set_title("DROWSY windows", color=DROWSY_COLOR,
                                       fontweight="bold")
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("steering")
        fig.suptitle("Example steering windows", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "02_example_windows.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  02_example_windows.png")

    # 3. Feature boxplots
    n_feat: int = len(feat_cols)
    n_rows_plot: int = (n_feat + 2) // 3
    fig, axes = plt.subplots(n_rows_plot, 3, figsize=(14, 3.5 * n_rows_plot))
    axes_flat = axes.flatten()
    for i, col in enumerate(feat_cols):
        alert_vals: FloatArray = out_df.loc[out_df["label"] == "alert", col].to_numpy()
        drowsy_vals: FloatArray = out_df.loc[out_df["label"] == "drowsy", col].to_numpy()
        bp = axes_flat[i].boxplot(
            [alert_vals, drowsy_vals], tick_labels=["alert", "drowsy"],
            patch_artist=True, widths=0.6,
            medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor(ALERT_COLOR + "99")
        bp["boxes"][1].set_facecolor(DROWSY_COLOR + "99")
        axes_flat[i].set_title(col, fontsize=10)
        axes_flat[i].grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Feature distributions: alert vs drowsy", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "03_feature_boxplots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  03_feature_boxplots.png")

    # 4. Correlation matrix
    corr: pd.DataFrame = out_df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_yticks(range(n_feat))
    ax.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feat_cols, fontsize=8)
    for r in range(n_feat):
        for c in range(n_feat):
            ax.text(c, r, f"{corr.iloc[r, c]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr.iloc[r, c]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "04_feature_correlation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  04_feature_correlation.png")

    # 5. Steering PSD
    def avg_psd(label: str) -> tuple[FloatArray | None, FloatArray | None]:
        psds: list[FloatArray] = []
        for _, l, _, ws in sessions:
            if l != label:
                continue
            for w in ws:
                nperseg: int = min(256, len(w))
                f: FloatArray
                pxx: FloatArray
                f, pxx = signal.welch(np.asarray(w, float), fs=fs_global,
                                      nperseg=nperseg)
                psds.append(pxx)
        if not psds:
            return None, None
        min_len: int = min(len(p) for p in psds)
        return f[:min_len], np.mean([p[:min_len] for p in psds], axis=0)

    f_a, psd_a = avg_psd("alert")
    f_d, psd_d = avg_psd("drowsy")
    if f_a is not None and f_d is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(f_a, psd_a, color=ALERT_COLOR, label="alert", lw=1.5)
        ax.semilogy(f_d, psd_d, color=DROWSY_COLOR, label="drowsy", lw=1.5)
        ax.axvline(lf_band[1], color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(lf_band[1] + 0.02, ax.get_ylim()[1] * 0.5,
                "LF / HF boundary", fontsize=8, color="gray")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (log scale)")
        ax.set_title("Average steering PSD: alert vs drowsy", fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "05_steering_psd.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  05_steering_psd.png")

    # 6. Per-session feature means
    session_means: pd.DataFrame = out_df.groupby(
        ["session", "label"])[feat_cols].mean()
    fig, axes = plt.subplots(n_rows_plot, 3, figsize=(14, 3.5 * n_rows_plot))
    axes_flat = axes.flatten()
    for i, col in enumerate(feat_cols):
        vals: pd.DataFrame = session_means[col].reset_index()
        colors: list[str] = [label_colors[l] for l in vals["label"]]
        axes_flat[i].barh(range(len(vals)), vals[col], color=colors)
        axes_flat[i].set_yticks(range(len(vals)))
        axes_flat[i].set_yticklabels(
            [s[:25] for s in vals["session"]], fontsize=7)
        axes_flat[i].set_title(col, fontsize=10)
        axes_flat[i].grid(axis="x", alpha=0.3)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Per-session feature means (blue=alert, red=drowsy)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "06_session_feature_means.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  06_session_feature_means.png")

    # 7. Steering velocity distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab, color in [("alert", ALERT_COLOR), ("drowsy", DROWSY_COLOR)]:
        all_vel: list[float] = []
        for _, l, _, ws in sessions:
            if l != lab:
                continue
            for w in ws:
                all_vel.extend(np.diff(w).tolist())
        if all_vel:
            ax.hist(all_vel, bins=100, alpha=0.5, color=color, label=lab,
                    density=True, range=(-0.3, 0.3))
    ax.set_xlabel("steering velocity (Δsteering per sample)")
    ax.set_ylabel("density")
    ax.set_title("Steering velocity distribution: alert vs drowsy\n"
                 "(heavier tails = overcorrections)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "07_steering_velocity_dist.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  07_steering_velocity_dist.png")

    print(f"\nAll plots saved to {plot_dir}/")

def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder of session CSVs")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--overlap", type=float, default=0.75)
    args: argparse.Namespace = ap.parse_args()

    paths: list[str] = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSVs found in {args.data}")

    # Pass 1: window every session
    sessions: list[tuple[str, str, int, list[FloatArray]]] = []
    fs_global: int = 20
    for p in paths:
        session, label, fs, windows = extract_session(
            p, args.window, args.overlap)
        sessions.append((session, label, fs, windows))
        fs_global = fs
        print(f"{session:40s} label={label:7s} windows={len(windows)}")

    # Alpha for steering entropy
    base_errs: FloatArray = _gather_prediction_errors(sessions, "alert")
    if len(base_errs) == 0:
        base_errs = _gather_prediction_errors(sessions, None)
        print("NOTE: no 'alert' sessions -> alpha from all data.")
    alpha: float = (
        float(np.percentile(np.abs(base_errs), 90))
        if len(base_errs) else float("nan")
    )
    print(f"steering-entropy alpha = {alpha:.5f}  (fs={fs_global} Hz)")

    # Pass 2: compute features
    rows: list[dict[str, Any]] = []
    for session, label, fs, windows in sessions:
        for w in windows:
            feat: dict[str, float] = compute_window_features(w, fs, alpha)
            feat["session"] = session
            feat["label"] = label
            feat["n_samples"] = len(w)
            rows.append(feat)

    out: pd.DataFrame = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No windows produced. Try shorter --window.")

    front: list[str] = ["session", "label", "n_samples"]
    feat_cols: list[str] = [c for c in out.columns if c not in front]
    out = out[front + feat_cols]

    before: int = len(out)
    out = out.dropna().reset_index(drop=True)
    if len(out) < before:
        print(f"dropped {before - len(out)} windows with NaN features")

    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} windows x {len(feat_cols)} features -> {args.out}")
    print("label counts:", out["label"].value_counts().to_dict())

    # Save extraction metadata
    meta: ExtractionMeta = ExtractionMeta(
        alpha=alpha, fs=fs_global,
        window_s=args.window, overlap=args.overlap)
    meta_path: str = args.out.replace(".csv", "_meta.json")
    with open(meta_path, "w") as f:
        f.write(meta.model_dump_json(indent=2))
    print(f"Saved extraction meta -> {meta_path}")

    # Plots
    _generate_plots(paths, sessions, out, feat_cols, fs_global)


if __name__ == "__main__":
    main()
