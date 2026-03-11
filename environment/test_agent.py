from stable_baselines3 import PPO
from environment import LoadBalancerEnv
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

# listas para grafico
hist_steps = []
hist_nodos = []
hist_reward = []
hist_cpu_total = [] 

for i in range(5000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    pesos_ruteo = action[:-1]       
    decision_escalado = action[-1]  
    activos = info['activos']
    
    # Sumamos el CPU de todos los nodos activos (índices 0, 6, 12... en el obs)
    cpu_total = 0.0
    for j in range(activos):
        cpu_total += obs[j * 6] 
        
    # Guardamos telemetría
    hist_steps.append(i)
    hist_nodos.append(activos)
    hist_reward.append(reward)
    hist_cpu_total.append(cpu_total)
    
    pesos_efectivos = np.zeros_like(pesos_ruteo)
    pesos_efectivos[:activos] = pesos_ruteo[:activos] 
    
    pesos_str = " ".join([f"{w:4.2f}" for w in pesos_efectivos])
    print(f"| {i+1:06d} | {activos:^7d} | {decision_escalado:^14.2f} | {reward:>9.2f} | [{pesos_str}] |")
        
    if terminated or truncated:
        print("-" * 110)
        motivo = "Colapso total (0 nodos vivos)" if terminated else "Limite de tiempo alcanzado"
        print(f"Terminating... Motivo: {motivo}")
        break

# --- GENERACIÓN DEL GRÁFICO FINAL ---
print("Generando gráfico de telemetría...")
fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje 1: Carga de CPU (Reflejo del tráfico)
color_cpu = 'tab:red'
ax1.set_xlabel('Pasos (Steps)')
ax1.set_ylabel('Carga Total de CPU (Tráfico)', color=color_cpu)
ax1.plot(hist_steps, hist_cpu_total, color=color_cpu, label='Carga Total del Sistema', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_cpu)

# Eje 2: Nodos Activos (Decisión del agente)
ax2 = ax1.twinx()  
color_nodos = 'tab:blue'
ax2.set_ylabel('Nodos Activos', color=color_nodos)
ax2.step(hist_steps, hist_nodos, color=color_nodos, label='Nodos Prendidos (Agente)', linewidth=2, where='post')
ax2.tick_params(axis='y', labelcolor=color_nodos)
ax2.set_ylim(0, 11) # Máximo 10 nodos

plt.title('Correlación: Tráfico Inyectado vs Reacción del Auto-Scaler (DRL)')
fig.tight_layout()

# Guardamos la imagen en la raíz del proyecto
plt.savefig('test_telemetria_agente.png', dpi=300)
print("¡Gráfico guardado exitosamente como 'test_telemetria_agente.png'!")