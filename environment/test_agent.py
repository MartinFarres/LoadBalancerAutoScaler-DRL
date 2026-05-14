from stable_baselines3 import PPO
from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_test_agent(nodes=5, iterations=5000, file='testing_metrics.csv'):
    
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    
    env = LoadBalancerEnv(simulated=True, max_steps=iterations, n_max=nodes, testing=True)

    print("Loading trained agent...")
    
    model = PPO.load(f"ppo_lb_production_ready_{nodes}_nodes")
    
    obs, info = env.reset(42)
    
    print("Begin traffic simulation...")
    print("-" * 110)
    print(f"| {'Step':^6} | {'Nodos':^7} | {'Scale Action':^14} | {'Reward':^9} | {'Pesos de Ruteo (HAProxy)':^45} |")
    print("-" * 110)
    
    # listas 
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
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        activos = info['activos']
        workload = info['workload'] * 4000 # Desnormalizamos -> asumiendo 4000 total_users
        
        
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
    formatted_file = f"test_ppo_{nodes}_nodes_i{iterations}_{file}"
    os.makedirs(f"./training_results/testing_results_{nodes}_nodes", exist_ok=True)
    save_path = os.path.join(f"./training_results/testing_results_{nodes}_nodes", formatted_file)
    pd.DataFrame({
        'cpu_promedio': hist_cpu_total,
        'ram_promedio': hist_ram_total,
        'latencia_promedio': hist_latency,
        'tasa_errores': hist_errors
    }).to_csv(save_path, index=False)
    print(f"Métricas del test guardadas en: {save_path}")

    # --- TABLA FINAL ---

    print("Generando tabla resumen...")
    viz = Visualizer()
    viz.generate_testing_summary_table(
        cpu_history=hist_cpu_total,
        ram_history=hist_ram_total,
        latency_history=hist_latency,
        errors_history=hist_errors,
        scaling_events=scaling_events,        
        sla_violation_pct=sla_violation_pct,  
        avg_cost_efficiency=avg_cost_efficiency
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='testing_metrics.csv')
    parser.add_argument('--iterations', type=int, default=5000)
    args = parser.parse_args()

    run_test_agent(nodes=args.nodes, iterations=args.iterations, file=args.file)
