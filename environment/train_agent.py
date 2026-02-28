import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv 
from typing import Callable

<<<<<<< Updated upstream
# Funcion para reducir el learning rate gradualmente (Learning Rate Schedule)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
=======
parser = argparse.ArgumentParser()
parser.add_argument("--simulated", action="store_true", help="Usa el modo simulado (matemático)")
args = parser.parse_args()

print(f"--- Iniciando entorno (Simulado: {args.simulated}) ---")
env = Monitor(LoadBalancerEnv(simulated=args.simulated))
>>>>>>> Stashed changes

env = Monitor(LoadBalancerEnv(simulated=False))
directory_logs = "./logs_tensorboard/"

print("Iniciando secuencia de entrenamiento con hiperparámetros optimizados...")

model = PPO("MlpPolicy", 
            env, 
            verbose=1, 
            n_steps=512,                  # Frecuencia de actualización
            batch_size=128,               # Tamaño del lote para promediar el ruido
            learning_rate=linear_schedule(0.0005), # Decaimiento lineal
            ent_coef=0.01,                # Coeficiente de entropía (Exploracion)
            tensorboard_log=directory_logs)

model.learn(total_timesteps=5000, tb_log_name="PPO_Tuned_5k") 

model.save("ppo_lb_agent_v1")
print("¡Entrenamiento completado y modelo guardado!")