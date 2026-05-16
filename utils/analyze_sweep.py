"""
Reads local W&B run data and regenerates all sweep analysis graphics in
resultados_graficos/sweep_analysis/.

Usage:
    python utils/analyze_sweep.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tensorboard.backend.event_processing import event_accumulator

WANDB_DIR = "./wandb"
OUT_DIR = "./resultados_graficos/sweep_analysis"
HYPERPARAMS = [
    "n_steps", "batch_size", "learning_rate",
    "clip_range", "gamma", "gae_lambda",
    "n_epochs", "vf_coef", "ent_coef",
]
HP_LABELS = {
    "n_steps": "n_steps",
    "batch_size": "batch_size",
    "learning_rate": "lr",
    "clip_range": "clip_range (ε)",
    "gamma": "γ",
    "gae_lambda": "λ_GAE",
    "n_epochs": "n_epochs",
    "vf_coef": "vf_coef (c₁)",
    "ent_coef": "ent_coef (c₂)",
}
TOP_N = 5


# ── data loading ──────────────────────────────────────────────────────────────

def load_runs() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(glob.glob(f"{WANDB_DIR}/run-*/")):
        run_id = run_dir.rstrip("/").split("/")[-1]
        cfg_path = os.path.join(run_dir, "files", "config.yaml")
        summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
        if not (os.path.exists(cfg_path) and os.path.exists(summary_path)):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        with open(summary_path) as f:
            summary = json.load(f)

        row = {"run_id": run_id}
        for hp in HYPERPARAMS:
            if hp in cfg:
                val = cfg[hp]
                row[hp] = val["value"] if isinstance(val, dict) and "value" in val else val
        row["ep_rew_mean"] = summary.get("rollout/ep_rew_mean")
        row["ep_len_mean"] = summary.get("rollout/ep_len_mean")
        row["value_loss"] = summary.get("train/value_loss")
        row["entropy_loss"] = summary.get("train/entropy_loss")
        row["explained_variance"] = summary.get("train/explained_variance")
        row["approx_kl"] = summary.get("train/approx_kl")
        row["clip_fraction"] = summary.get("train/clip_fraction")
        rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=["ep_rew_mean"])
    df = df.sort_values("ep_rew_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def load_learning_curve(run_id: str) -> pd.DataFrame | None:
    short_id = run_id.split("-")[-1]
    tb_base = glob.glob(
        f"{WANDB_DIR}/{run_id}/files/logs_tensorboard/sweep_{short_id}/PPO_1"
    )
    if not tb_base:
        return None
    ea = event_accumulator.EventAccumulator(tb_base[0])
    ea.Reload()
    if "rollout/ep_rew_mean" not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars("rollout/ep_rew_mean")
    return pd.DataFrame({"step": [e.step for e in events], "reward": [e.value for e in events]})


# ── plot helpers ──────────────────────────────────────────────────────────────

PALETTE = "#2196F3"
TOP_COLOR = "#FF5722"
CMAP = "RdYlGn"


def reward_cmap(values):
    norm = Normalize(vmin=min(values), vmax=max(values))
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    return [sm.to_rgba(v) for v in values]


# ── figure 1 – ranked bar chart ───────────────────────────────────────────────

def plot_ranked_runs(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = reward_cmap(df["ep_rew_mean"].values)
    bars = ax.barh(
        range(len(df)), df["ep_rew_mean"].values,
        color=colors, edgecolor="none", height=0.7
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"#{r}  {rid.split('-')[-1]}" for r, rid in zip(df["rank"], df["run_id"])],
        fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean Episode Reward (ep_rew_mean)", fontsize=11)
    ax.set_title("All Sweep Runs – Ranked by Final Reward", fontsize=13, fontweight="bold")
    ax.axvline(df["ep_rew_mean"].iloc[0], color=TOP_COLOR, lw=1.5, ls="--", label=f"Best: {df['ep_rew_mean'].iloc[0]:.1f}")
    ax.axvline(df["ep_rew_mean"].mean(), color="grey", lw=1, ls=":", label=f"Mean: {df['ep_rew_mean'].mean():.1f}")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    sm = ScalarMappable(norm=Normalize(vmin=df["ep_rew_mean"].min(), vmax=df["ep_rew_mean"].max()), cmap=CMAP)
    fig.colorbar(sm, ax=ax, label="ep_rew_mean", shrink=0.6)
    fig.tight_layout()
    _save(fig, "1_ranked_runs.png")


# ── figure 2 – scatter grid ───────────────────────────────────────────────────

def plot_scatter_grid(df: pd.DataFrame):
    hps = [h for h in HYPERPARAMS if h in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(hps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    colors = reward_cmap(df["ep_rew_mean"].values)

    for i, hp in enumerate(hps):
        ax = axes[i]
        ax.scatter(df[hp], df["ep_rew_mean"], c=colors, s=60, edgecolors="white", linewidths=0.5, zorder=3)
        # highlight top run
        best = df.iloc[0]
        ax.scatter(best[hp], best["ep_rew_mean"], c=TOP_COLOR, s=120, marker="*", zorder=5, label="Best")
        ax.set_xlabel(HP_LABELS.get(hp, hp), fontsize=10)
        ax.set_ylabel("ep_rew_mean" if i % ncols == 0 else "", fontsize=9)
        ax.grid(alpha=0.3)
        if hp == "learning_rate":
            ax.set_xscale("log")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Hyperparameter vs Final Reward (Scatter Grid)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "2_scatter_grid.png")


# ── figure 3 – correlation heatmap ───────────────────────────────────────────

def plot_correlations(df: pd.DataFrame):
    hps = [h for h in HYPERPARAMS if h in df.columns]
    corr = df[hps + ["ep_rew_mean"]].corr()["ep_rew_mean"].drop("ep_rew_mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CMAP_pos if v >= 0 else CMAP_neg for v in corr.values]

    def color_for(v):
        if v >= 0:
            return plt.cm.Greens(min(v * 2 + 0.3, 1.0))
        else:
            return plt.cm.Reds(min(-v * 2 + 0.3, 1.0))

    bar_colors = [color_for(v) for v in corr.values]
    bars = ax.barh(
        [HP_LABELS.get(h, h) for h in corr.index],
        corr.values,
        color=bar_colors,
        edgecolor="none",
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r with ep_rew_mean", fontsize=11)
    ax.set_title("Hyperparameter Correlation with Final Reward", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, corr.values):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, "3_correlations.png")

CMAP_pos = None
CMAP_neg = None


# ── figure 4 – parallel coordinates ──────────────────────────────────────────

def plot_parallel_coordinates(df: pd.DataFrame):
    hps = [h for h in HYPERPARAMS if h in df.columns]
    fig, ax = plt.subplots(figsize=(14, 6))

    df_norm = df[hps].copy()
    for hp in hps:
        mn, mx = df_norm[hp].min(), df_norm[hp].max()
        df_norm[hp] = (df_norm[hp] - mn) / (mx - mn + 1e-12)

    rew_norm = (df["ep_rew_mean"] - df["ep_rew_mean"].min()) / (df["ep_rew_mean"].max() - df["ep_rew_mean"].min() + 1e-12)
    cmap = plt.cm.RdYlGn

    for idx, row in df_norm.iterrows():
        color = cmap(rew_norm.iloc[idx])
        lw = 2.5 if idx == 0 else 0.6
        alpha = 1.0 if idx == 0 else 0.4
        ax.plot(range(len(hps)), row[hps].values, color=color, lw=lw, alpha=alpha, zorder=3 if idx == 0 else 2)

    ax.set_xticks(range(len(hps)))
    ax.set_xticklabels([HP_LABELS.get(h, h) for h in hps], fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Normalized value", fontsize=10)
    ax.set_title("Parallel Coordinates – All Runs (colored by reward)", fontsize=13, fontweight="bold")

    sm = ScalarMappable(norm=Normalize(vmin=df["ep_rew_mean"].min(), vmax=df["ep_rew_mean"].max()), cmap=cmap)
    fig.colorbar(sm, ax=ax, label="ep_rew_mean", shrink=0.7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "4_parallel_coordinates.png")


# ── figure 5 – top-N table ────────────────────────────────────────────────────

def plot_top5_table(df: pd.DataFrame):
    top = df.head(TOP_N).copy()
    hps_show = ["n_steps", "batch_size", "learning_rate", "clip_range", "gamma", "gae_lambda", "n_epochs", "vf_coef"]
    col_labels = ["Run ID", "Reward"] + [HP_LABELS.get(h, h) for h in hps_show]

    table_data = []
    for _, row in top.iterrows():
        short_id = row["run_id"].split("-")[-1]
        lr_str = f"{row['learning_rate']:.2e}" if "learning_rate" in row else "—"
        cells = [short_id, f"{row['ep_rew_mean']:.1f}"]
        for hp in hps_show:
            val = row.get(hp, "—")
            if hp == "learning_rate":
                cells.append(f"{val:.2e}")
            elif isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(int(val)) if not pd.isna(val) else "—")
        table_data.append(cells)

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    # header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # row alternating + best highlight
    for i in range(1, TOP_N + 1):
        color = "#E8F5E9" if i == 1 else ("#F5F5F5" if i % 2 == 0 else "white")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    ax.set_title(f"Top {TOP_N} Runs – Hyperparameter Summary", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "5_top5_table.png")


# ── figure 6 – learning curves for top-N ─────────────────────────────────────

def plot_learning_curves(df: pd.DataFrame):
    top = df.head(TOP_N)
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.cm.tab10
    for i, (_, row) in enumerate(top.iterrows()):
        curve = load_learning_curve(row["run_id"])
        if curve is None:
            continue
        short_id = row["run_id"].split("-")[-1]
        label = f"#{row['rank']}  {short_id}  (final={row['ep_rew_mean']:.0f})"
        lw = 2.5 if i == 0 else 1.5
        ax.plot(curve["step"], curve["reward"], lw=lw, label=label, color=cmap(i))

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("ep_rew_mean", fontsize=11)
    ax.set_title(f"Learning Curves – Top {TOP_N} Runs", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "6_learning_curves_top5.png")


# ── util ──────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_best(df: pd.DataFrame):
    best = df.iloc[0]
    print("\n=== Best run ===")
    print(f"  Run ID      : {best['run_id']}")
    print(f"  ep_rew_mean : {best['ep_rew_mean']:.4f}")
    print(f"\n  Hyperparameters:")
    for hp in HYPERPARAMS:
        val = best.get(hp, "—")
        if hp == "learning_rate":
            print(f"    {hp:20s}: {val:.6e}")
        elif isinstance(val, float):
            print(f"    {hp:20s}: {val:.8f}")
        else:
            print(f"    {hp:20s}: {val}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading W&B run data...")
    df = load_runs()
    print(f"  {len(df)} runs loaded.")
    print_best(df)

    print("\nGenerating plots...")
    plot_ranked_runs(df)
    plot_scatter_grid(df)
    plot_correlations(df)
    plot_parallel_coordinates(df)
    plot_top5_table(df)
    plot_learning_curves(df)
    print("\nDone. All plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
