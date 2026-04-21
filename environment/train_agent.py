from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv
from callbacks import TrainingMetricsCallback
from visualizer import Visualizer
import sys
import os

directory_logs = "./logs_tensorboard/"
MODEL_PATH = "ppo_lb_simulated_base"

def train_phase_1_simulation():
    print("Iniciando entrenamiento en Simulacion Pura...")
    
    env_sim = Monitor(LoadBalancerEnv(simulated=True))
    
    model = PPO("MlpPolicy", 
                env_sim, 
                verbose=1, 
                n_steps=512,                  
                batch_size=128,               
                learning_rate=0.0005, # Fijo para la simulación
                ent_coef=0.01,                
                tensorboard_log=directory_logs)

    metrics_callback = TrainingMetricsCallback(save_dir="./training_results/phase1")

    model.learn(total_timesteps=3000000, tb_log_name="PPO_Phase1_Simulated", callback=metrics_callback) 

    model.save(MODEL_PATH)
    print("Fase 1 completada. Conocimiento base guardado.\n")

    print("Generando curva de aprendizaje para Fase 1...")
    viz = Visualizer(save_dir="./resultados_graficos/phase1")
    viz.plot_learning_curve("./training_results/phase1/training_metrics.csv")


def train_phase_2_real_world():
    print("Iniciando entrenamiento con docker + HAProxy...")
    
    env_real = Monitor(LoadBalancerEnv(simulated=False))
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print("No se encontró el modelo base simulado.")
        return

    model = PPO.load(MODEL_PATH, env=env_real, tensorboard_log=directory_logs)
    
    model.learning_rate = 0.0001

    metrics_callback_real = TrainingMetricsCallback(save_dir="./training_results/phase2")

    model.learn(total_timesteps=15000, tb_log_name="PPO_Phase2_Real_FineTuned", callback=metrics_callback_real)

    model.save("ppo_lb_production_ready")
    print("Fase 2 completada.\n")

    print("Generando curva de aprendizaje para Fase 2...")
    viz = Visualizer(save_dir="./resultados_graficos/phase2")
    viz.plot_learning_curve("./training_results/phase2/training_metrics.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        comando = sys.argv[1]
        
        if comando == "train_phase_1_simulation":
            train_phase_1_simulation()
        elif comando == "train_phase_2_real_world":
            train_phase_2_real_world()
        else:
            print(f"Comando desconocido: {comando}")
    else:
        print("Por favor, especifica la fase a entrenar.")