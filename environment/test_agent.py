from stable_baselines3 import PPO
from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=120)

env = LoadBalancerEnv(simulated=True, max_steps=5000)

print("Loading trained agent...")

model = PPO.load("ppo_lb_simulated_base")

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

for i in range(5000):
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