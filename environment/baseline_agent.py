import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import time
import pandas as pd
from utils.config import TOTAL_USERS

def run_industry_baseline(simulated=True, steps=5000, n_max=5, file='testing_metrics.csv'):
    print("Iniciando prueba del Baseline de la Industria (Round Robin + Thresholds)...")
    
    env = LoadBalancerEnv(simulated=simulated, max_steps=steps, n_max=n_max, testing=True)
    obs, info = env.reset(42)
    
    # Auto Scaler Tradicional
    CPU_THRESHOLD_UP = 0.75   # Escalar si CPU > 75%
    CPU_THRESHOLD_DOWN = 0.25 # Desescalar si CPU < 25%
    
    hist_reward = []
    hist_cpu_total = []
    hist_ram_total = []
    hist_latency = []
    hist_errors = []
    hist_workload = []
    hist_activos = []

     # metricas
    scaling_events = 0
    sla_violations = 0
    last_activos = info.get('activos', 1)


    # Inicializamos el visualizador
    viz = Visualizer(save_dir="./resultados_graficos/baseline")

    for i in range(steps):
        # ── Action from pre-step obs ──────────────────────────────────────
        activos_pre = info.get('activos', 1)

        cpu_for_action = 0.0
        for j in range(activos_pre):
            cpu_for_action += obs[j * 6]
        avg_cpu_action = cpu_for_action / activos_pre if activos_pre > 0 else 0.0

        scale_decision = 0.5
        if avg_cpu_action >= CPU_THRESHOLD_UP:
            scale_decision = 1.0  # Scale Up
        elif avg_cpu_action <= CPU_THRESHOLD_DOWN:
            scale_decision = 0.0  # Scale Down

        weights = [1.0] * env.n_max
        action = np.array(weights + [scale_decision], dtype=np.float32)

        # ── Step ─────────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)

        # ── Log from post-step state ──────────────────────────────────────
        activos = info.get('activos', 1)
        workload = info.get('workload', 0.0) * TOTAL_USERS

        if i > 0 and activos != last_activos:
            scaling_events += 1
        last_activos = activos

        cpu_total = 0.0
        ram_total = 0.0
        avg_latency = 0.0
        total_errors = 0.0

        for j in range(activos):
            cpu_total   += obs[j * 6]      # CPU
            ram_total   += obs[j * 6 + 1]  # RAM
            avg_latency += obs[j * 6 + 3]  # Latency
            total_errors+= obs[j * 6 + 4]  # Error Rate

        avg_cpu = cpu_total / activos if activos > 0 else 0.0

        if activos > 0:
            cpu_total   /= activos
            ram_total   /= activos
            avg_latency /= activos

        if avg_latency > 0.5:
            sla_violations += 1

        hist_reward.append(reward)
        hist_cpu_total.append(avg_cpu)
        hist_ram_total.append(ram_total)
        hist_latency.append(avg_latency)
        hist_errors.append(total_errors)
        hist_activos.append(activos)
        hist_workload.append(workload)

        if terminated or truncated:
            break
    
    total_steps = len(hist_cpu_total)
    sla_violation_pct = (sla_violations / total_steps) * 100

    # Eficiencia de costo: Usuarios soportados por cada contenedor activo
    cost_efficiencies = [w/max(1, a) for w, a in zip(hist_workload, hist_activos)]
    avg_cost_efficiency = np.mean(cost_efficiencies)
    
        
    mode_tag = "sim" if simulated else "real"
    formatted_file = f"test_bai_{mode_tag}_n{n_max}_i{steps}_{file}"
    save_dir = f"./training_results/testing_results_{n_max}_nodes"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, formatted_file)
    pd.DataFrame({
        'step': range(total_steps),
        'reward': hist_reward,
        'cpu_mean': hist_cpu_total,
        'ram_mean': hist_ram_total,
        'latency_mean': hist_latency,
        'error_mean': hist_errors,
        'workload': [w / TOTAL_USERS for w in hist_workload],
        'activos': hist_activos
    }).to_csv(save_path, index=False)
    print(f"Métricas del BAI guardadas en: {save_path}")

    print("Generando tabla resumen del Baseline BAI...")
    viz.generate_testing_summary_table(
        cpu_history=hist_cpu_total,
        ram_history=hist_ram_total,
        latency_history=hist_latency,
        errors_history=hist_errors,
        scaling_events=scaling_events,
        sla_violation_pct=sla_violation_pct,
        avg_cost_efficiency=avg_cost_efficiency,
        mode=mode_tag,
    )

    print("Generando curva de recompensa del Baseline BAI...")
    viz.plot_testing_reward_curve(csv_path=save_path, mode=mode_tag)

    print("Generando gráficos de comportamiento de workload del Baseline BAI...")
    viz.plot_testing_behavior(csv_path=save_path, n_max=n_max, mode=mode_tag)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='testing_metrics.csv')
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--simulated', action='store_true', default=False)
    args = parser.parse_args()

    run_industry_baseline(simulated=args.simulated, steps=args.iterations, n_max=args.nodes, file=args.file)
