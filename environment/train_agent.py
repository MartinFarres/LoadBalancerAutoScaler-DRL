from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv
import sys
import os

directory_logs = "./logs_tensorboard/"
MODEL_PATH = "ppo_lb_simulated_base"

def train_phase_1_simulation():
    print("Iniciando entrenamiento en Simulacion Pura)...")
    
    env_sim = Monitor(LoadBalancerEnv(simulated=True))
    
    model = PPO("MlpPolicy", 
                env_sim, 
                verbose=1, 
                n_steps=512,                  
                batch_size=128,               
                learning_rate=0.0005, # Fijo para la simulación
                ent_coef=0.01,                
                tensorboard_log=directory_logs)

    model.learn(total_timesteps=300000, tb_log_name="PPO_Phase1_Simulated") 

    model.save(MODEL_PATH)
    print("Fase 1 completada. Conocimiento base guardado.\n")

def train_phase_2_real_world():
    print("Iniciando entrenamiento con docker + HAProxy)...")
    
    env_real = Monitor(LoadBalancerEnv(simulated=False))
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print("No se encontró el modelo base simulado.")
        return

    model = PPO.load(MODEL_PATH, env=env_real, tensorboard_log=directory_logs)
    
    # Reducimos el learning rate para que no "olvide" lo aprendido de golpe,
    # solo queremos que haga un ajuste fino (fine-tuning) al ruido de la red real.
    model.learning_rate = 0.0001

    # total_timesteps debe ser un multiplo o por lo menos mayor que n_steps
    model.learn(total_timesteps=5000, tb_log_name="PPO_Phase2_Real_FineTuned")

    model.save("ppo_lb_production_ready")
    print("Fase 2 completada")



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