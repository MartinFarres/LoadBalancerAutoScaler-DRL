"""
Analisis de sensibilidad global (muestreo de Saltelli + indices de Sobol) sobre los pesos de la
funcion de recompensa de LoadBalancerEnv (ver environment/environment.py::reward_function), en dos
etapas. Ver SALTELLI_SENSITIVITY.md (raiz del repo) para la explicacion completa del metodo y de
cada decision de diseño tomada aca.

Etapa 1 (subcomando `sensitivity`): para cada combinacion de pesos generada por Saltelli, entrena un
PPO liviano desde cero sobre el entorno simulado con esos pesos, evalua la politica resultante en
una corrida corta y determinista, y reduce el resultado a 5 KPIs operativos (latencia, error rate,
SLA violations, costo, eventos de escalado) a UN SOLO tamaño de flota. El analisis de Sobol sobre
esas corridas indica que peso influye mas en cada KPI.

Etapa 2 (subcomando `robust-search`): a partir de los pesos identificados como influyentes en la
Etapa 1 (ver `select_influential_weights`), busca la combinacion que mejor generaliza a traves de
VARIOS tamaños de flota (`FLEET_SIZES`), agregando cada KPI por tamaño con una formula que penaliza
tanto el promedio como la inconsistencia entre tamaños, y rankeando las combinaciones por un score
compuesto (promedio normalizado de los 5 KPIs robustos).

Usage:
    # Etapa 1: sensibilidad a un tamaño de flota fijo
    python utils/sensitivity_analysis.py sensitivity --n 8 --nodes 5 --train-timesteps 20000 --eval-steps 2000

    # Smoke test rapido antes de una corrida larga:
    python utils/sensitivity_analysis.py sensitivity --n 2 --train-timesteps 500 --eval-steps 200 --no-second-order

    # Etapa 2: busqueda robusta multi-escala, solo sobre los pesos influyentes segun la Etapa 1
    python utils/sensitivity_analysis.py robust-search --active-weights W_ERRORS,W_LATENCY,W_QUEUE --n 8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# environment/ no tiene __init__.py: los scripts que viven ahi (train_agent.py, test_agent.py)
# importan entre si porque Python agrega automaticamente el directorio del script en ejecucion a
# sys.path. Un modulo en utils/ no recibe ese path gratis, asi que lo agregamos explicitamente.
sys.path.append(os.path.join(ROOT_DIR, "environment"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils.config import REWARD_WEIGHTS, REWARD_WEIGHT_BOUNDS, SEED, USERS_PER_NODE
from environment import LoadBalancerEnv
from test_agent import rollout_and_collect
from train_agent import linear_schedule

KPI_NAMES = ["avg_latency", "avg_error_rate", "sla_violation_pct", "avg_active_ratio", "scaling_events"]

# Tamaños de flota representativos para la Etapa 2 (busqueda robusta multi-escala). Cada tamaño
# implica un mini-entrenamiento propio: n_max cambia la forma de la observacion (n_max*6+1) y de la
# accion (n_max+1), asi que un PPO entrenado para un tamaño no sirve para otro.
FLEET_SIZES = [5, 10, 15, 20, 25]

# Mismos hiperparametros del mejor run de W&B usados en train_phase_1_simulation (ver CLAUDE.md
# "Key Hyperparameters (Best W&B Run)"). Se mantienen fijos en todas las muestras a proposito: solo
# reward_weights y train_timesteps varian entre corridas, para que cualquier diferencia en los KPIs
# sea atribuible a los pesos muestreados y no a una deriva de hiperparametros de PPO.
PPO_HYPERPARAMS = dict(
    n_steps=2048,
    batch_size=64,
    clip_range=0.1,
    vf_coef=0.75,
    gamma=0.91,
    ent_coef=0.0001,
    gae_lambda=0.947,
    n_epochs=6,
    normalize_advantage=True,
)


def build_problem(weight_names: list = None) -> dict:
    """
    Arma el 'problem' spec de SALib a partir de REWARD_WEIGHT_BOUNDS. Si `weight_names` se pasa,
    restringe el espacio de muestreo a ese subconjunto (usado en la Etapa 2 para variar solo los
    pesos identificados como influyentes en la Etapa 1, dejando el resto en su default).
    """
    names = weight_names if weight_names is not None else list(REWARD_WEIGHT_BOUNDS.keys())
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": [REWARD_WEIGHT_BOUNDS[name] for name in names],
    }


def generate_saltelli_samples(problem: dict, n_base_samples: int, calc_second_order: bool = True,
                               seed: int = SEED) -> np.ndarray:
    """
    Genera la matriz de muestras de Saltelli. Con D = problem['num_vars'] variables, devuelve
    N*(2D+2) filas (esquema de segundo orden) o N*(D+2) filas (calc_second_order=False), cada una
    un vector de D valores de pesos dentro de sus bounds.
    """
    return sobol_sample.sample(problem, n_base_samples, calc_second_order=calc_second_order, seed=seed)


def _row_to_weights(problem: dict, row: np.ndarray) -> dict:
    """
    Convierte una fila muestreada en un dict {nombre_peso: valor} listo para LoadBalancerEnv,
    completando con los defaults de REWARD_WEIGHTS los pesos que no forman parte de `problem`
    (Etapa 2, cuando `problem` solo cubre un subconjunto de pesos influyentes).
    """
    weights = dict(REWARD_WEIGHTS)
    weights.update({name: float(value) for name, value in zip(problem["names"], row)})
    return weights


def evaluate_weight_sample(weights: dict, nodes: int = 5, train_timesteps: int = 20000,
                            eval_steps: int = 2000, seed: int = SEED, total_users: int = None) -> dict:
    """
    Mini-entrena un PPO desde cero bajo `weights` y evalua la politica resultante, devolviendo 5
    KPIs operativos escalares. No se usa la recompensa total como salida: al depender linealmente
    de los propios pesos que se estan variando, cualquier diferencia seria en gran parte tautologica
    (ver SALTELLI_SENSITIVITY.md).

    `total_users`: si se pasa, sobreescribe utils.config.TOTAL_USERS para esta corrida (usado por
    evaluate_weight_sample_multiscale para asignarle a cada tamano de flota el total_users que le
    corresponde). Si se deja en None, usa el default de LoadBalancerEnv (utils.config.TOTAL_USERS).
    """
    # --- Entrenamiento: instancia propia, nunca reutilizada para evaluar ---
    raw_train_env = Monitor(LoadBalancerEnv(simulated=True, n_max=nodes, reward_weights=weights,
                                             total_users=total_users))
    vec_env = DummyVecEnv([lambda: raw_train_env])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        norm_env,
        verbose=0,
        learning_rate=linear_schedule(1.94e-03),
        seed=seed,
        device="cpu",
        **PPO_HYPERPARAMS,
    )
    model.learn(total_timesteps=train_timesteps)

    # Congelamos la normalizacion para que la evaluacion use estadisticas fijas, no siga actualizandolas.
    norm_env.training = False
    norm_env.norm_reward = False

    # --- Evaluacion: instancia NUEVA, reseteada una sola vez con `seed` fijo. LoadBalancerEnv solo
    # siembra su reloj de trafico simulado en el primer reset() de cada instancia (ver
    # environment.py::reset); reutilizar raw_train_env aca mezclaria la fase de trafico ya avanzada
    # por el entrenamiento con la de otras muestras, contaminando la comparacion entre pesos.
    raw_eval_env = LoadBalancerEnv(simulated=True, n_max=nodes, reward_weights=weights, testing=True,
                                    total_users=total_users)
    result = rollout_and_collect(raw_eval_env, model, obs_normalizer=norm_env, iterations=eval_steps, seed=seed)

    total_steps = max(1, result["total_steps"])
    avg_active_ratio = float(np.mean([a / nodes for a in result["hist_activos"]])) if result["hist_activos"] else 0.0

    return {
        "avg_latency": float(np.mean(result["hist_latency"])) if result["hist_latency"] else 0.0,
        "avg_error_rate": float(np.mean(result["hist_errors"])) if result["hist_errors"] else 0.0,
        "sla_violation_pct": result["sla_violations"] / total_steps * 100,
        "avg_active_ratio": avg_active_ratio,
        "scaling_events": float(result["scaling_events"]),
    }


def plot_run_comparison(samples_df: pd.DataFrame, problem: dict, figures_dir: str,
                         n_base_samples: int, kpi_names: list = None,
                         color_by: str = "avg_latency") -> list:
    """
    Genera diagramas que comparan el rendimiento (KPIs) de cada corrida con los pesos que la
    generaron, complementando los indices de Sobol agregados (que resumen sensibilidad promedio,
    pero no muestran corrida por corrida como se relacionan pesos y desempeño):

      1. Scatter grid por KPI: un subplot por peso (peso en x, KPI en y), un punto por corrida.
      2. Coordenadas paralelas: una linea por corrida a lo largo de los 8 pesos + los KPIs (todos
         normalizados min-max a [0,1] para poder compararlos en la misma escala), coloreada segun
         `color_by` para poder rastrear que combinaciones de pesos llevan a mejor/peor desempeño en
         ese KPI de referencia.

    Devuelve la lista de paths de las figuras generadas.
    """
    kpi_names = kpi_names or KPI_NAMES
    weight_names = problem["names"]
    figure_paths = []

    # --- 1. Scatter grid: peso (x) vs KPI (y), un punto por corrida ---
    ncols = 4
    nrows = int(np.ceil(len(weight_names) / ncols))
    for kpi in kpi_names:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
        for idx, weight_name in enumerate(weight_names):
            ax = axes[idx // ncols][idx % ncols]
            ax.scatter(samples_df[weight_name], samples_df[kpi], s=18, alpha=0.7, color="#4C72B0")
            ax.set_xlabel(weight_name, fontsize=9)
            if idx % ncols == 0:
                ax.set_ylabel(kpi, fontsize=9)
            ax.tick_params(labelsize=8)
        for idx in range(len(weight_names), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig.suptitle(f"Rendimiento por corrida: {kpi} vs. cada peso", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = os.path.join(figures_dir, f"run_comparison_scatter_{kpi}_n{n_base_samples}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figure_paths.append(fig_path)

    # --- 2. Coordenadas paralelas: una linea por corrida, ejes = pesos + KPIs normalizados ---
    axes_cols = weight_names + kpi_names
    norm_df = pd.DataFrame(index=samples_df.index)
    for col in axes_cols:
        values = samples_df[col].to_numpy(dtype=float)
        col_min, col_max = values.min(), values.max()
        norm_df[col] = 0.5 if np.isclose(col_max, col_min) else (values - col_min) / (col_max - col_min)

    color_values = samples_df[color_by].to_numpy(dtype=float)
    if np.isclose(color_values.max(), color_values.min()):
        color_norm = np.full_like(color_values, 0.5)
    else:
        color_norm = (color_values - color_values.min()) / (color_values.max() - color_values.min())

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(1.4 * len(axes_cols), 6))
    x_positions = np.arange(len(axes_cols))
    for row_idx in range(len(norm_df)):
        y_values = norm_df.iloc[row_idx][axes_cols].to_numpy(dtype=float)
        ax.plot(x_positions, y_values, color=cmap(color_norm[row_idx]), alpha=0.6, linewidth=1.2)

    # Linea divisoria entre el bloque de pesos (entrada) y el bloque de KPIs (salida)
    ax.axvline(x=len(weight_names) - 0.5, linestyle="--", color="gray", alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axes_cols, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Valor normalizado [0, 1]")
    ax.set_title(f"Comparacion de corridas: pesos vs. KPIs (color = {color_by})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_values.min(), vmax=color_values.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=color_by)
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, f"run_comparison_parallel_coords_n{n_base_samples}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figure_paths.append(fig_path)

    return figure_paths


def _samples_path(output_dir: str, n_base_samples: int, nodes: int, train_timesteps: int) -> str:
    return os.path.join(output_dir, f"samples_kpis_n{n_base_samples}_nodes{nodes}_tt{train_timesteps}.csv")


def _summary_path(output_dir: str, n_base_samples: int, nodes: int, train_timesteps: int) -> str:
    return os.path.join(output_dir, f"sobol_indices_summary_n{n_base_samples}_nodes{nodes}_tt{train_timesteps}.csv")


def run_sensitivity_analysis(n_base_samples: int = 8, nodes: int = 5, train_timesteps: int = 20000,
                              eval_steps: int = 2000, calc_second_order: bool = True, seed: int = SEED,
                              output_dir: str = "./training_results/sensitivity_analysis",
                              figures_dir: str = "./resultados_graficos/sensitivity_analysis") -> dict:
    """
    Orquesta el pipeline completo: muestreo Saltelli -> mini-entrenamiento+evaluacion por muestra ->
    analisis de Sobol por KPI. Guarda un CSV incremental (checkpoint despues de cada muestra, asi una
    corrida larga corrida en background puede interrumpirse y resumirse sin recalcular lo ya hecho),
    un CSV resumen de los indices de Sobol, un bar chart de sensibilidad (ST) por KPI, y los diagramas
    de comparacion corrida-por-corrida de `plot_run_comparison` (scatter peso-vs-KPI + coordenadas
    paralelas).
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    problem = build_problem()
    X = generate_saltelli_samples(problem, n_base_samples, calc_second_order=calc_second_order, seed=seed)

    samples_path = _samples_path(output_dir, n_base_samples, nodes, train_timesteps)
    summary_path = _summary_path(output_dir, n_base_samples, nodes, train_timesteps)

    if os.path.exists(samples_path):
        rows = pd.read_csv(samples_path).to_dict("records")
        done_ids = {int(r["sample_id"]) for r in rows}
    else:
        rows = []
        done_ids = set()

    total = len(X)
    for i, row in enumerate(X):
        if i in done_ids:
            continue
        weights = _row_to_weights(problem, row)
        print(f"[sensitivity_analysis] Muestra {i + 1}/{total}: {weights}")
        kpis = evaluate_weight_sample(weights, nodes=nodes, train_timesteps=train_timesteps,
                                       eval_steps=eval_steps, seed=seed)
        rows.append({"sample_id": i, **weights, **kpis})
        rows.sort(key=lambda r: r["sample_id"])
        pd.DataFrame(rows).to_csv(samples_path, index=False)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    summary_rows = []
    for kpi in KPI_NAMES:
        Y = df[kpi].to_numpy(dtype=float)
        # SALib.analyze.sobol necesita varianza en Y: con muestras/pasos muy chicos (smoke tests) o
        # un KPI que no se mueve entre corridas (ej. scaling_events=0 en todas), Y sale constante y
        # la libreria revienta en vez de devolver indices indefinidos. Lo detectamos antes y
        # completamos con NaN para ese KPI, sin abortar el resto del analisis.
        if np.isclose(Y.std(), 0.0):
            print(f"[sensitivity_analysis] AVISO: '{kpi}' salio constante en todas las muestras "
                  "(Y.std()==0) — no se puede calcular sensibilidad, se completa con NaN. "
                  "Probablemente train_timesteps/eval_steps son muy chicos para este KPI.")
            for name in problem["names"]:
                summary_rows.append({
                    "kpi": kpi, "weight_name": name,
                    "S1": np.nan, "S1_conf": np.nan, "ST": np.nan, "ST_conf": np.nan,
                })
            continue

        Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, seed=seed,
                                    print_to_console=False)
        for j, name in enumerate(problem["names"]):
            summary_rows.append({
                "kpi": kpi,
                "weight_name": name,
                "S1": Si["S1"][j],
                "S1_conf": Si["S1_conf"][j],
                "ST": Si["ST"][j],
                "ST_conf": Si["ST_conf"][j],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    figure_paths = []
    for kpi in KPI_NAMES:
        kpi_df = summary_df[summary_df["kpi"] == kpi].sort_values("ST", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(kpi_df["weight_name"], kpi_df["ST"], yerr=kpi_df["ST_conf"], capsize=4, color="#4C72B0")
        ax.set_ylabel("Indice de Sobol de orden total (ST)")
        ax.set_title(f"Sensibilidad de los pesos de recompensa sobre {kpi}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f"sobol_ST_{kpi}_n{n_base_samples}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figure_paths.append(fig_path)

    figure_paths.extend(plot_run_comparison(df, problem, figures_dir, n_base_samples))

    print(f"[sensitivity_analysis] Muestras + KPIs: {samples_path}")
    print(f"[sensitivity_analysis] Indices de Sobol: {summary_path}")
    print(f"[sensitivity_analysis] Figuras: {figures_dir}")

    return {
        "samples_df": df,
        "sobol_summary_df": summary_df,
        "samples_path": samples_path,
        "summary_path": summary_path,
        "figure_paths": figure_paths,
    }


def select_influential_weights(sobol_summary_df: pd.DataFrame, top_k: int = 4, metric: str = "ST") -> list:
    """
    A partir del resumen de indices de Sobol de la Etapa 1 (un run a un solo tamaño de flota),
    selecciona los `top_k` pesos con mayor `metric` promedio entre los 5 KPIs (por defecto ST,
    sensibilidad de orden total). Los KPIs que salieron constantes en la Etapa 1 (NaN, ver nota de
    robustez en run_sensitivity_analysis) se ignoran automaticamente al promediar. Los pesos NO
    seleccionados quedan congelados en su default de REWARD_WEIGHTS para la Etapa 2.
    """
    ranking = sobol_summary_df.groupby("weight_name")[metric].mean().sort_values(ascending=False)
    return ranking.head(top_k).index.tolist()


def evaluate_weight_sample_multiscale(weights: dict, fleet_sizes: list = None, train_timesteps: int = 20000,
                                       eval_steps: int = 2000, seed: int = SEED,
                                       robustness_lambda: float = 1.0, users_per_node: float = None) -> dict:
    """
    Corre evaluate_weight_sample(weights, nodes=n, ...) para cada tamaño de flota en `fleet_sizes`
    (cada uno un mini-entrenamiento independiente: n_max cambia la forma de obs/accion, un PPO
    entrenado para un tamaño no sirve para otro). A cada tamaño se le asigna
    total_users = users_per_node * n (default utils.config.USERS_PER_NODE, 9 usuarios/nodo, calibrado
    para que la relacion de headroom n_max*NODE_CAPACITY/total_users sea igual en todos los tamaños —
    sin esto, tamaños grandes quedarian sobredimensionados frente a un workload fijo y los KPIs de
    costo/escalado saldrian sesgados). Para cada uno de los 5 KPIs, agrega los valores obtenidos en
    los distintos tamaños con:

        robust_kpi = mean(valores_por_tamaño) + robustness_lambda * (max(valores) - min(valores))

    Esto premia una combinacion de pesos que en promedio funciona bien Y que ademas es consistente
    entre tamaños de flota (el segundo termino penaliza la dispersion/inconsistencia). Devuelve un
    dict con los 5 KPIs robustos (prefijo `robust_`) y ademas los valores crudos por tamaño
    (`{kpi}_n{tamaño}`) para poder inspeccionar/graficar el detalle despues.
    """
    fleet_sizes = fleet_sizes or FLEET_SIZES
    users_per_node = users_per_node if users_per_node is not None else USERS_PER_NODE
    per_size_kpis = {
        n: evaluate_weight_sample(weights, nodes=n, train_timesteps=train_timesteps,
                                   eval_steps=eval_steps, seed=seed,
                                   total_users=round(users_per_node * n))
        for n in fleet_sizes
    }

    result = {}
    for kpi in KPI_NAMES:
        values = np.array([per_size_kpis[n][kpi] for n in fleet_sizes], dtype=float)
        result[f"robust_{kpi}"] = float(values.mean() + robustness_lambda * (values.max() - values.min()))
        for n in fleet_sizes:
            result[f"{kpi}_n{n}"] = per_size_kpis[n][kpi]

    return result


def _robust_search_path(output_dir: str, n_base_samples: int, active_weight_names: list,
                         fleet_sizes: list, train_timesteps: int, robustness_lambda: float,
                         users_per_node: float) -> str:
    weights_tag = "-".join(active_weight_names)
    sizes_tag = "-".join(str(n) for n in fleet_sizes)
    return os.path.join(
        output_dir,
        f"robust_search_n{n_base_samples}_w[{weights_tag}]_sizes[{sizes_tag}]_tt{train_timesteps}_lam{robustness_lambda}_upn{users_per_node}.csv",
    )


def run_robust_weight_search(active_weight_names: list, n_base_samples: int = 8, fleet_sizes: list = None,
                              train_timesteps: int = 20000, eval_steps: int = 2000,
                              robustness_lambda: float = 1.0, users_per_node: float = None,
                              calc_second_order: bool = False, seed: int = SEED,
                              output_dir: str = "./training_results/robust_weight_search",
                              figures_dir: str = "./resultados_graficos/robust_weight_search") -> dict:
    """
    Etapa 2: busca, entre los pesos en `active_weight_names` (tipicamente los que
    `select_influential_weights` identifico como influyentes en la Etapa 1), la combinacion que
    mejor generaliza a traves de `fleet_sizes`. Los pesos que NO estan en `active_weight_names`
    quedan fijos en su default de REWARD_WEIGHTS (ver `_row_to_weights`).

    Reutiliza el generador de muestras de Saltelli (build_problem/generate_saltelli_samples)
    restringido a las dimensiones de `active_weight_names` como diseño espacio-llenador para
    generar candidatos bien distribuidos — en esta etapa no se llama a SALib.analyze (no se
    calculan indices de Sobol aca, el objetivo es encontrar el mejor punto, no la sensibilidad),
    por lo que por defecto se usa el esquema mas barato (`calc_second_order=False`).

    Por cada muestra: evaluate_weight_sample_multiscale(...) -> 5 KPIs robustos. Se normalizan
    (min-max) los 5 KPIs robustos entre todas las muestras y se promedian con igual peso en un
    `composite_score` (menor = mejor, ya que los 5 KPIs son "badness"). Guarda un CSV incremental
    (mismo patron de checkpointing/resumability que la Etapa 1) y un CSV ordenado por
    composite_score con la mejor combinacion arriba.
    """
    fleet_sizes = fleet_sizes or FLEET_SIZES
    users_per_node = users_per_node if users_per_node is not None else USERS_PER_NODE
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    problem = build_problem(active_weight_names)
    X = generate_saltelli_samples(problem, n_base_samples, calc_second_order=calc_second_order, seed=seed)

    samples_path = _robust_search_path(output_dir, n_base_samples, active_weight_names, fleet_sizes,
                                        train_timesteps, robustness_lambda, users_per_node)

    if os.path.exists(samples_path):
        rows = pd.read_csv(samples_path).to_dict("records")
        done_ids = {int(r["sample_id"]) for r in rows}
    else:
        rows = []
        done_ids = set()

    total = len(X)
    for i, row in enumerate(X):
        if i in done_ids:
            continue
        weights = _row_to_weights(problem, row)
        print(f"[robust_weight_search] Muestra {i + 1}/{total} (pesos activos: {active_weight_names}): "
              f"{ {k: weights[k] for k in active_weight_names} }")
        result = evaluate_weight_sample_multiscale(weights, fleet_sizes=fleet_sizes,
                                                     train_timesteps=train_timesteps, eval_steps=eval_steps,
                                                     seed=seed, robustness_lambda=robustness_lambda,
                                                     users_per_node=users_per_node)
        rows.append({"sample_id": i, **weights, **result})
        rows.sort(key=lambda r: r["sample_id"])
        pd.DataFrame(rows).to_csv(samples_path, index=False)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    robust_cols = [f"robust_{kpi}" for kpi in KPI_NAMES]
    norm = df[robust_cols].copy()
    for col in robust_cols:
        col_min, col_max = norm[col].min(), norm[col].max()
        norm[col] = 0.5 if np.isclose(col_max, col_min) else (norm[col] - col_min) / (col_max - col_min)
    # Score compuesto = promedio con igual peso de los 5 KPIs robustos normalizados; menor = mejor
    # (son todos KPIs de "badness": latencia, error rate, SLA%, costo, scaling events).
    df["composite_score"] = norm.mean(axis=1)

    df = df.sort_values("composite_score").reset_index(drop=True)
    ranked_path = samples_path.replace(".csv", "_ranked.csv")
    df.to_csv(ranked_path, index=False)

    best = df.iloc[0].to_dict()
    best_weights = dict(REWARD_WEIGHTS)
    best_weights.update({name: best[name] for name in active_weight_names})

    print(f"[robust_weight_search] Muestras + KPIs (ranking): {ranked_path}")
    print(f"[robust_weight_search] Mejor combinacion (composite_score={best['composite_score']:.4f}):")
    for name in active_weight_names:
        print(f"    {name} = {best[name]:.3f}")

    return {
        "ranked_df": df,
        "samples_path": samples_path,
        "ranked_path": ranked_path,
        "best_weights": best_weights,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisis de sensibilidad Saltelli/Sobol de los pesos de la funcion de recompensa (LBDRL)."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sens_parser = subparsers.add_parser(
        "sensitivity",
        help="Etapa 1: indices de Sobol de los 8 pesos a un tamaño de flota fijo.",
    )
    sens_parser.add_argument("--n", type=int, default=8,
                              help="Tamaño base N del muestreo de Saltelli; total de corridas = N*(2D+2), D=8 pesos.")
    sens_parser.add_argument("--nodes", type=int, default=5)
    sens_parser.add_argument("--train-timesteps", type=int, default=20000)
    sens_parser.add_argument("--eval-steps", type=int, default=2000)
    sens_parser.add_argument("--no-second-order", action="store_true",
                              help="Usa el esquema mas barato N*(D+2) (sin indices S2 de interaccion).")
    sens_parser.add_argument("--output-dir", type=str, default="./training_results/sensitivity_analysis")
    sens_parser.add_argument("--figures-dir", type=str, default="./resultados_graficos/sensitivity_analysis")
    sens_parser.add_argument("--seed", type=int, default=None)

    robust_parser = subparsers.add_parser(
        "robust-search",
        help="Etapa 2: busca la combinacion de pesos que mejor generaliza a traves de varios tamaños de flota.",
    )
    robust_parser.add_argument("--active-weights", type=str, required=True,
                                help="Lista separada por comas de los pesos a variar (ej. "
                                     "W_ERRORS,W_LATENCY,W_QUEUE), tipicamente los que salieron "
                                     "influyentes en `sensitivity`. El resto queda fijo en su default.")
    robust_parser.add_argument("--n", type=int, default=8,
                                help="Tamaño base N del muestreo de Saltelli sobre los pesos activos.")
    robust_parser.add_argument("--fleet-sizes", type=str, default=None,
                                help="Lista separada por comas de tamaños de flota (default: "
                                     f"{','.join(str(n) for n in FLEET_SIZES)}).")
    robust_parser.add_argument("--train-timesteps", type=int, default=20000)
    robust_parser.add_argument("--eval-steps", type=int, default=2000)
    robust_parser.add_argument("--robustness-lambda", type=float, default=1.0,
                                help="Peso de la penalizacion por inconsistencia entre tamaños de "
                                     "flota en robust_kpi = mean + lambda*(max-min).")
    robust_parser.add_argument("--users-per-node", type=float, default=None,
                                help="Usuarios simulados por nodo asignados a cada tamaño de flota "
                                     f"(total_users = users_per_node * n_max; default utils.config.USERS_PER_NODE={USERS_PER_NODE}).")
    robust_parser.add_argument("--second-order", action="store_true",
                                help="Usa el esquema N*(2D+2) (por defecto se usa el mas barato N*(D+2), "
                                     "ya que esta etapa no calcula indices de Sobol).")
    robust_parser.add_argument("--output-dir", type=str, default="./training_results/robust_weight_search")
    robust_parser.add_argument("--figures-dir", type=str, default="./resultados_graficos/robust_weight_search")
    robust_parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "sensitivity":
        run_sensitivity_analysis(
            n_base_samples=args.n,
            nodes=args.nodes,
            train_timesteps=args.train_timesteps,
            eval_steps=args.eval_steps,
            calc_second_order=not args.no_second_order,
            seed=args.seed if args.seed is not None else SEED,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )
    elif args.mode == "robust-search":
        active_weights = [w.strip() for w in args.active_weights.split(",") if w.strip()]
        unknown = set(active_weights) - set(REWARD_WEIGHTS)
        if unknown:
            parser.error(f"--active-weights contiene pesos desconocidos: {unknown}")
        fleet_sizes = ([int(n.strip()) for n in args.fleet_sizes.split(",") if n.strip()]
                        if args.fleet_sizes else None)

        run_robust_weight_search(
            active_weight_names=active_weights,
            n_base_samples=args.n,
            fleet_sizes=fleet_sizes,
            train_timesteps=args.train_timesteps,
            eval_steps=args.eval_steps,
            robustness_lambda=args.robustness_lambda,
            users_per_node=args.users_per_node,
            calc_second_order=args.second_order,
            seed=args.seed if args.seed is not None else SEED,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )
