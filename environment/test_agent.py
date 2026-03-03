from stable_baselines3 import PPO
from environment import LoadBalancerEnv
import numpy as np

env = LoadBalancerEnv()

print("Loading trained agent...")
model = PPO.load("ppo_lb_production_ready")

obs, info = env.reset()

print("Begin traffic simmulation...")

for i in range(501):
    # deterministic = True, para que el modelo prediga la mejor opcion | Testing Phase    
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 10 == 0:
        pesos_ruteo = action[:-1]       # Los primeros 10 valores
        decision_escalado = action[-1]  # El último valor
        
        print(f"\n--- Instance {i+1} ---")
        print(f"Weight decision (Proxy): {pesos_ruteo}")
        print(f"Auto-scaling decision: {decision_escalado} (<0.3 Baja, >0.7 Sube)")
        print(f"Reward: {reward:.2f}")
        print(f"Active Containers: {info['activos']}")
        
    if terminated or truncated:
        print("Terminating...")
        break