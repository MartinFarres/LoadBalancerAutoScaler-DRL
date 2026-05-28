import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import pandas as pd
from utils.config import TOTAL_USERS, SEED

def run_test_agent(nodes=5, iterations=5000, file='testing_metrics.csv', simulated=True):

    np.set_printoptions(precision=2, suppress=True, linewidth=120)

    raw_env = LoadBalancerEnv(simulated=simulated, max_steps=iterations, n_max=nodes, testing=True)

    # Load the VecNormalize statistics saved at the end of the matching training phase.
    # The policy network was trained on VecNormalize-normalised observations, so we must
    # apply the same transform at inference time or the action distribution will be wrong.
    if simulated:
        pkl_path = f"./training_results/phase1_{nodes}_nodes/vec_normalize_phase1.pkl"
    else:
        pkl_path = f"./training_results/phase2_{nodes}_nodes/vec_normalize_phase2.pkl"

    obs_normalizer = None
    if os.path.exists(pkl_path):
        _dummy = DummyVecEnv([lambda: raw_env])
        obs_normalizer = VecNormalize.load(pkl_path, _dummy)
        obs_normalizer.training = False   # freeze running statistics
        obs_normalizer.norm_reward = False
        print(f"VecNormalize statistics loaded from: {pkl_path}")
    else:
        print(f"Warning: {pkl_path} not found — running without obs normalisation.")

    print("Loading trained agent...")

    model = PPO.load(f"ppo_lb_production_ready_{nodes}_nodes")

    obs, info = raw_env.reset(SEED)

    print("Begin traffic simulation...")
    print("-" * 110)
    print(f"| {'Step':^6} | {'Nodos':^7} | {'Scale Action':^14} | {'Reward':^9} | {'Pesos de Ruteo (HAProxy)':^45} |")
    print("-" * 110)

    # listas
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
    last_activos = 1

    for i in range(iterations):
        obs_for_predict = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
        action, _states = model.predict(obs_for_predict, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action)
        
        activos = info['activos']
        workload_norm = info['workload']
        workload = workload_norm * TOTAL_USERS

        hist_reward.append(reward)

        # Conteo de eventos de escalado
        if i > 0 and activos != last_activos:
            scaling_events += 1
        last_activos = activos
        
        cpu_total = 0.0
        ram_total = 0.0
        avg_latency = 0.0
        total_errors = 0.0
        
        for j in range(activos):
            cpu_total += obs[j * 6]      # CPU
            ram_total += obs[j * 6 + 1]  #  RAM (% de uso)
            avg_latency += obs[j * 6 + 3] #  Latency
            total_errors += obs[j * 6 + 4] # Error Rate
            
        if activos > 0:
            cpu_total /= activos  # Sacamos el promedio real
            ram_total /= activos  # Sacamos el promedio real 
            avg_latency /= activos
        
        # Conteo de violaciones SLA (latencia > 0.5 )
        if avg_latency > 0.5:
            sla_violations += 1
            
        # Guardamos
        hist_cpu_total.append(cpu_total)
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
    
    # Guardar a CSV
    mode_tag = "sim" if simulated else "real"
    formatted_file = f"test_ppo_{mode_tag}_{nodes}_nodes_i{iterations}_{file}"
    save_dir = f"./training_results/testing_results_{nodes}_nodes"
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
    print(f"Métricas del test guardadas en: {save_path}")

    # --- GRAFICOS Y TABLA FINAL ---

    viz = Visualizer(save_dir="./resultados_graficos/ppo")

    print("Generando tabla resumen...")
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

    print("Generando curva de recompensa...")
    viz.plot_testing_reward_curve(csv_path=save_path, mode=mode_tag)

    print("Generando gráficos de comportamiento de workload...")
    viz.plot_testing_behavior(csv_path=save_path, n_max=nodes, mode=mode_tag)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='testing_metrics.csv')
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--simulated', action='store_true', default=False)
    args = parser.parse_args()

    run_test_agent(nodes=args.nodes, iterations=args.iterations, file=args.file, simulated=args.simulated)
