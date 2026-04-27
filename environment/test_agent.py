from stable_baselines3 import PPO
from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_test_agent(nodes=5, iterations=5000, file='testing_metrics.csv'):
    
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    
    env = LoadBalancerEnv(simulated=True, max_steps=iterations, n_max=nodes)
    
    print("Loading trained agent...")
    
    model = PPO.load("ppo_lb_production_ready")
    
    obs, info = env.reset()
    
    print("Begin traffic simulation...")
    print("-" * 110)
    print(f"| {'Step':^6} | {'Nodos':^7} | {'Scale Action':^14} | {'Reward':^9} | {'Pesos de Ruteo (HAProxy)':^45} |")
    print("-" * 110)
    
    # listas 
    hist_cpu_total = [] 
    hist_ram_total = [] 
    hist_latency = []   
    hist_errors = []  
    
    for i in range(iterations):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        activos = info['activos']
        
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
            
        # Guardamos
        hist_cpu_total.append(cpu_total)
        hist_ram_total.append(ram_total)
        hist_latency.append(avg_latency)
        hist_errors.append(total_errors)
            
        if terminated or truncated:
            break
    
    
    # Guardar a CSV
    formatted_file = f"test_ppo_n{nodes}_i{iterations}_{file}"
    save_path = os.path.join("./training_results/phase2", formatted_file) # or somewhere else for test
    os.makedirs("./training_results/testing_results", exist_ok=True)
    save_path = os.path.join("./training_results/testing_results", formatted_file)
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
        high_latency_threshold=0.8 
    )
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='testing_metrics.csv')
    parser.add_argument('--iterations', type=int, default=5000)
    args = parser.parse_args()

    run_test_agent(nodes=args.nodes, iterations=args.iterations, file=args.file)
