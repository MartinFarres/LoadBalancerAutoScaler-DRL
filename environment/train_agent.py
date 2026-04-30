import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv
from callbacks import TrainingMetricsCallback
from visualizer import Visualizer
import sys
from typing import Callable
import torch
import os
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

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

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


directory_logs = "./logs_tensorboard/"
MODEL_PATH = "ppo_lb_simulated_base"

def train_phase_1_simulation(nodes=5, iterations=50000, file="training_metrics.csv"):
    print(f"Iniciando entrenamiento en Simulacion Pura para {iterations} pasos con {nodes} nodos...")
    
    env_sim = Monitor(LoadBalancerEnv(simulated=True, n_max=nodes))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PPO("MlpPolicy", 
            env_sim, 
            verbose=1, 
            n_steps=2048,           # Ganador indiscutible en la grafica de lineas
            batch_size=128,         # Buen equilibrio segun coordenadas paralelas
            learning_rate=0.0003,   # Bajo, debido a la fuerte correlacion negativa
            clip_range=0.3,         # El que mas rapido convergio
            vf_coef=0.5,            # Vital: Evita el colapso mostrado en el scatter plot
            gamma=0.95,             # Las lineas amarillas pasaban por este rango inferior
            ent_coef=0.01,          # Irrelevante segun el scatter plot, se deja bajo
            tensorboard_log=directory_logs,
            device=device)

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PPO.load(MODEL_PATH, env=env_real, tensorboard_log=directory_logs, device=device)
    
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

def run_wandb_sweep(nodes=5, iterations=100000):
    print("Iniciando W&B Sweep para optimización de hiperparámetros...")
    
    sweep_config = {
        'method': 'bayes', # Optimizacion Bayesiana (encuentra el óptimo más rápido que Grid o Random)
        'metric': {
            'name': 'rollout/ep_rew_mean', # Métrica a optimizar
            'goal': 'maximize'   
        },
        'parameters': {
            
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 3e-3},
            'gamma': {'min': 0.85, 'max': 0.999},

            # Mem y Lotes
            'n_steps': {'values': [128, 256, 512, 1024, 2048]},
            'batch_size': {'values': [64, 128, 256]},
            'n_epochs': {'values': [3,6,10,15,20]},
            
            #PPO specific
            'clip_range': {'values': [0.1, 0.2, 0.3]},              # Épsilon (ε)
            'ent_coef': {'values': [0.0, 0.0001, 0.001, 0.01]},     # c2 (Coeficiente de entropía)
            'vf_coef': {'values': [0.5, 0.75, 1.0]},                # c1 (Coeficiente de valor)
            'gae_lambda': {'min': 0.9, 'max': 1.0},                 # λ_GAE
            'target_kl': {'min': 0.003, 'max': 0.03},               # Límite de divergencia KL
            'normalize_advantage': {'values': [True, False]}        # Normalización
        }
    }

    def sweep_train():
        run = wandb.init(sync_tensorboard=True)
        config = wandb.config

        # batch_size debe ser factor de n_steps y menor o igual
        if config.batch_size > config.n_steps:
            config.batch_size = config.n_steps

        env_sim = Monitor(LoadBalancerEnv(simulated=True, n_max=nodes))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = PPO("MlpPolicy", 
                    env_sim, 
                    verbose=0, 
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    n_steps=config.n_steps,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    clip_range=config.clip_range,
                    ent_coef=config.ent_coef,
                    vf_coef=config.vf_coef,
                    gae_lambda=config.gae_lambda,
                    target_kl=config.target_kl,
                    normalize_advantage=config.normalize_advantage,
                    tensorboard_log=f"./logs_tensorboard/sweep_{run.id}",
                    device=device)

        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/sweep_{run.id}",
            verbose=2
        )

        model.learn(total_timesteps=iterations, callback=wandb_callback)
        # Cerramos el run
        run.finish()

    sweep_id = wandb.sweep(sweep_config, project="LoadBalancerAutoScaler-DRL")
    
    wandb.agent(sweep_id, sweep_train, count=40)


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
    elif args.comando == "sweep":
        run_wandb_sweep(nodes=args.nodes, iterations=args.iterations)
