from stable_baselines3 import PPO
from environment import LoadBalancerEnv
import numpy as np

# Set Numpy para imprimir con 2 decimales y sin saltos de linea
np.set_printoptions(precision=2, suppress=True, linewidth=120)

env = LoadBalancerEnv(simulated=False)

print("Loading trained agent...")
model = PPO.load("ppo_lb_production_ready")

obs, info = env.reset()

print("Begin traffic simulation...")
print("-" * 110)
print(f"| {'Step':^6} | {'Nodos':^7} | {'Scale Action':^14} | {'Reward':^9} | {'Pesos de Ruteo (HAProxy)':^45} |")
print("-" * 110)

for i in range(500):
   # deterministic = True, para que el modelo prediga la mejor opcion
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    pesos_ruteo = action[:-1]       
    decision_escalado = action[-1]  
    
    activos = info['activos']
    pesos_efectivos = np.zeros_like(pesos_ruteo)
    pesos_efectivos[:activos] = pesos_ruteo[:activos] 
    
    pesos_str = " ".join([f"{w:4.2f}" for w in pesos_efectivos])
    print(f"| {i+1:06d} | {activos:^7d} | {decision_escalado:^14.2f} | {reward:>9.2f} | [{pesos_str}] |")
        
    if terminated or truncated:
        print("-" * 110)
        motivo = "Colapso total (0 nodos vivos)" if terminated else "Limite de tiempo alcanzado"
        print(f"Terminating... Motivo: {motivo}")
        break