"""Generate the two optional figures proposed for section 4 of the report.

1. Overlaid Phase-1 learning curves (rollout/ep_rew_mean) for N in {5,10,15,20,25}.
2. Pareto scatter `cost vs error_rate` per (agent, N) in sim and real.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training_results"
OUTPUT_DIR = REPO_ROOT / "resultados_graficos" / "discussion_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_SIZES = [5, 10, 15, 20, 25]
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
AGENT_COLORS = {"PPO": "#0072B2", "BAI": "#D55E00", "PID": "#009E73"}
AGENT_MARKERS = {"PPO": "o", "BAI": "s", "PID": "^"}

SIM_ITERS = 250_000
REAL_ITERS = 2_000


def apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_phase1_overlay() -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for color, n in zip(OKABE_ITO, NODE_SIZES):
        csv = TRAINING_DIR / f"phase1_{n}_nodes" / "training_metrics.csv"
        df = pd.read_csv(csv)
        df = df.dropna(subset=["rollout/ep_rew_mean"])
        x = df["timestep"].to_numpy() / 1e3
        y = df["rollout/ep_rew_mean"].to_numpy()
        smoothed = pd.Series(y).rolling(window=10, min_periods=1).mean().to_numpy()
        ax.plot(x, smoothed, color=color, label=f"N = {n}", alpha=0.95)

    ax.set_xlabel("Pasos de entrenamiento (×10³)")
    ax.set_ylabel(r"Recompensa media por episodio  $\overline{R}_{\mathrm{ep}}$")
    ax.set_title("Convergencia de Fase 1 según tamaño de cluster", pad=8)
    ax.axhline(0.0, color="0.7", lw=0.6, ls=":", zorder=0)
    ax.legend(title="Tamaño", frameon=False, loc="lower right", ncol=2)
    ax.grid(alpha=0.25, lw=0.4)

    out = OUTPUT_DIR / "phase1_learning_curves_overlay.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _load_test(agent: str, mode: str, n: int, iters: int) -> pd.DataFrame:
    if agent == "ppo":
        name = f"test_ppo_{mode}_{n}_nodes_i{iters}_training_metrics.csv"
    else:
        name = f"test_{agent}_{mode}_n{n}_i{iters}_training_metrics.csv"
    csv = TRAINING_DIR / f"testing_results_{n}_nodes" / name
    return pd.read_csv(csv)


def _aggregate(agent_label: str, agent_key: str, mode: str, iters: int) -> pd.DataFrame:
    rows = []
    for n in NODE_SIZES:
        df = _load_test(agent_key, mode, n, iters)
        rows.append(
            {
                "agent": agent_label,
                "N": n,
                "cost": (df["activos"] / n).mean(),
                "error_rate": df["error_mean"].mean(),
                "latency": df["latency_mean"].mean(),
                "reward": df["reward"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_pareto_scatter() -> Path:
    sim = pd.concat(
        [
            _aggregate("PPO", "ppo", "sim", SIM_ITERS),
            _aggregate("BAI", "bai", "sim", SIM_ITERS),
            _aggregate("PID", "pid", "sim", SIM_ITERS),
        ]
    )
    real = pd.concat(
        [
            _aggregate("PPO", "ppo", "real", REAL_ITERS),
            _aggregate("BAI", "bai", "real", REAL_ITERS),
            _aggregate("PID", "pid", "real", REAL_ITERS),
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=False)
    size_scale = {n: 35 + 14 * i for i, n in enumerate(NODE_SIZES)}

    for ax, data, title in zip(axes, [sim, real], ["Simulación", "Cluster real"]):
        for agent, sub in data.groupby("agent"):
            ax.scatter(
                sub["cost"],
                sub["error_rate"],
                s=[size_scale[n] for n in sub["N"]],
                color=AGENT_COLORS[agent],
                marker=AGENT_MARKERS[agent],
                edgecolors="white",
                linewidths=0.7,
                label=agent,
                alpha=0.9,
                zorder=3,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    f"N={int(row['N'])}",
                    (row["cost"], row["error_rate"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=6.5,
                    color="0.25",
                )
        ax.set_xlabel(r"Costo  $\overline{a}/N$")
        ax.set_title(title, pad=6)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=-0.005)

    axes[0].set_ylabel("Tasa de errores  $\\overline{e}$")
    axes[1].legend(title="Agente", frameon=False, loc="upper right")
    fig.suptitle(
        "Frente de Pareto: costo de aprovisionamiento vs. tasa de errores",
        y=1.02,
        fontsize=10,
    )

    out = OUTPUT_DIR / "pareto_cost_vs_error.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _build_compare_df() -> pd.DataFrame:
    rows = []
    for n in NODE_SIZES:
        for mode, iters in [("sim", SIM_ITERS), ("real", REAL_ITERS)]:
            for label, key in [("PPO", "ppo"), ("BAI", "bai"), ("PID", "pid")]:
                df = _load_test(key, mode, n, iters)
                rows.append(
                    {
                        "agent": label,
                        "N": n,
                        "mode": mode,
                        "reward": df["reward"].mean(),
                        "cpu": df["cpu_mean"].mean(),
                        "lat": df["latency_mean"].mean(),
                        "err": df["error_mean"].mean(),
                        "cost": (df["activos"] / n).mean(),
                    }
                )
    return pd.DataFrame(rows)


def plot_reward_bars() -> Path:
    data = _build_compare_df()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=False)
    agents = ["PPO", "BAI", "PID"]
    width = 0.26
    x = np.arange(len(NODE_SIZES))

    for ax, mode, title in zip(axes, ["sim", "real"], ["Simulación", "Cluster real"]):
        sub = data[data["mode"] == mode]
        for i, agent in enumerate(agents):
            vals = [sub[(sub["agent"] == agent) & (sub["N"] == n)]["reward"].iloc[0] for n in NODE_SIZES]
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                color=AGENT_COLORS[agent],
                label=agent,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in NODE_SIZES])
        ax.set_title(title, pad=6)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.grid(axis="y", alpha=0.25, lw=0.4)

    axes[0].set_ylabel(r"Recompensa media  $\overline{R}$")
    axes[1].legend(title="Agente", frameon=False, loc="lower right")
    fig.suptitle("Recompensa media por agente y tamaño de cluster", y=1.02, fontsize=10)

    out = OUTPUT_DIR / "reward_bars_by_agent_N.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_metric_diff_bars() -> Path:
    data = _build_compare_df()
    metrics = [("lat", "Latencia"), ("err", "Errores"), ("cost", "Costo")]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)

    for ax, mode, title in zip(axes, ["sim", "real"], ["Simulación", "Cluster real"]):
        sub = data[data["mode"] == mode]
        diffs = {"BAI": [], "PID": []}
        for n in NODE_SIZES:
            ppo = sub[(sub["agent"] == "PPO") & (sub["N"] == n)].iloc[0]
            for base in ["BAI", "PID"]:
                row = sub[(sub["agent"] == base) & (sub["N"] == n)].iloc[0]
                metric_vals = []
                for key, _ in metrics:
                    base_val = row[key]
                    ppo_val = ppo[key]
                    if base_val <= 1e-9:
                        metric_vals.append(0.0)
                    else:
                        metric_vals.append(100.0 * (base_val - ppo_val) / base_val)
                diffs[base].append(metric_vals)

        n_metrics = len(metrics)
        width = 0.36
        x = np.arange(n_metrics)
        for i, base in enumerate(["BAI", "PID"]):
            arr = np.array(diffs[base]).mean(axis=0)
            ax.bar(
                x + (i - 0.5) * width,
                arr,
                width,
                color=AGENT_COLORS[base],
                label=f"vs. {base}",
                edgecolor="white",
                linewidth=0.6,
            )
            for xi, val in zip(x + (i - 0.5) * width, arr):
                ax.annotate(
                    f"{val:+.0f}%",
                    (xi, val),
                    textcoords="offset points",
                    xytext=(0, 4 if val >= 0 else -10),
                    ha="center",
                    fontsize=7,
                    color="0.2",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics])
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title(title, pad=6)
        ax.grid(axis="y", alpha=0.25, lw=0.4)

    axes[0].set_ylabel(r"Mejora de PPO  $(\%)$")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Mejora porcentual de PPO sobre cada baseline (promediada sobre N)",
        y=1.02,
        fontsize=10,
    )

    out = OUTPUT_DIR / "ppo_improvement_over_baselines.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    apply_pub_style()
    f1 = plot_phase1_overlay()
    f2 = plot_pareto_scatter()
    f3 = plot_reward_bars()
    f4 = plot_metric_diff_bars()
    for p in (f1, f2, f3, f4):
        print(f"Saved: {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
