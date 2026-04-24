import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv
from callbacks import TrainingMetricsCallback
from visualizer import Visualizer
import sys
import os
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class StepLoggerCallback(BaseCallback):
    """
    Callback personalizado para imprimir el progreso paso a paso 
    durante el lento entrenamiento en el entorno real.
    """
    def _on_step(self) -> bool:
        # Extraemos la recompensa del paso actual
        reward = self.locals.get('rewards', [0.0])[0]
        
        # Extraemos la información del entorno (nodos activos)
        infos = self.locals.get('infos', [{}])
        activos = infos[0].get('activos', 0)
        
        # Imprimimos una línea limpia por cada paso
        print(f"-> [Real Training] Step: {self.num_timesteps:04d} | Nodos Activos: {activos} | Reward: {reward:>7.2f}")
        
        return True


directory_logs = "./logs_tensorboard/"
MODEL_PATH = "./logs_checkpoints/ppo_real_env_2000_steps"

def train_phase_1_simulation(nodes=5, iterations=50000, file="training_metrics.csv"):
    print(f"Iniciando entrenamiento en Simulacion Pura para {iterations} pasos con {nodes} nodos...")
    
    env_sim = Monitor(LoadBalancerEnv(simulated=True, n_max=nodes))
    
    model = PPO("MlpPolicy", 
                env_sim, 
                verbose=1, 
                n_steps=512,                  
                batch_size=128,               
                learning_rate=0.001, # Fijo para la simulación
                ent_coef=0.01,                
                tensorboard_log=directory_logs)

    metrics_callback = TrainingMetricsCallback(save_dir="./training_results/phase1", file_name=file)

    model.learn(total_timesteps=iterations, tb_log_name="PPO_Phase1_Simulated", callback=metrics_callback) 

    model.save(MODEL_PATH)
    print("Fase 1 completada. Conocimiento base guardado.\n")

    print("Generando curva de aprendizaje para Fase 1...")
    viz = Visualizer(save_dir="./resultados_graficos/phase1")
    viz.plot_learning_curve(f"./training_results/phase1/{file}")


def train_phase_2_real_world(nodes=5, iterations=5000, file="training_metrics.csv"):
    print(f"Iniciando entrenamiento con docker + HAProxy para {iterations} pasos con {nodes} nodos...")
    
    env_real = Monitor(LoadBalancerEnv(simulated=False, n_max=nodes))
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print("No se encontró el modelo base simulado.")
        return

    model = PPO.load(MODEL_PATH, env=env_real, tensorboard_log=directory_logs)
    
    model.verbose = 1
    model.learning_rate = 0.0001

    metrics_callback_real = TrainingMetricsCallback(save_dir="./training_results/phase2", file_name=file)
    logger_callback = StepLoggerCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,                  
        save_path='./logs_checkpoints/',
        name_prefix='ppo_real_env'       
    )


    model.learn(total_timesteps=iterations, tb_log_name="PPO_Phase2_Real_FineTuned", callback=[metrics_callback_real, logger_callback, checkpoint_callback])

    model.save("ppo_lb_production_ready")
    print("Fase 2 completada.\n")

    print("Generando curva de aprendizaje para Fase 2...")
    viz = Visualizer(save_dir="./resultados_graficos/phase2")
    viz.plot_learning_curve(f"./training_results/phase2/{file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('comando', type=str)
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='training_metrics.csv')
    parser.add_argument('--iterations', type=int, default=50000)
    args = parser.parse_args()

    if args.comando == "train_phase_1_simulation":
        train_phase_1_simulation(nodes=args.nodes, iterations=args.iterations, file=args.file)
    elif args.comando == "train_phase_2_real_world":
        train_phase_2_real_world(nodes=args.nodes, iterations=args.iterations, file=args.file)
