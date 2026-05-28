import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv
from callbacks import TrainingMetricsCallback, WorkloadBehaviorCallback
from visualizer import Visualizer
import sys
from typing import Callable
import torch
import os
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer
import wandb
from wandb.integration.sb3 import WandbCallback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SEED

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
def model_path(nodes):
    return f"ppo_lb_simulated_base_{nodes}_nodes"

def train_phase_1_simulation(nodes=5, iterations=500000, file="training_metrics.csv"):
    print(f"Iniciando entrenamiento en Simulacion Pura para {iterations} pasos con {nodes} nodos...")
    
    raw_env = Monitor(LoadBalancerEnv(simulated=True, n_max=nodes))
    vec_env = DummyVecEnv([lambda: raw_env])
    
    # Aplicamos VecNormalize al entorno vectorizado
    env_sim = VecNormalize(
        vec_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0, 
        clip_reward=10.0
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hiperparámetros del mejor run del W&B Bayesian Sweep (ID: 75589c1o)
    # reward=-217 vs. promedio de -345 en los otros 28 runs
    model = PPO("MlpPolicy",
            env_sim,
            verbose=1,
            n_steps=128,
            batch_size=64,
            learning_rate=linear_schedule(1.13e-03),
            clip_range=0.3,
            vf_coef=0.5,
            gamma=0.8622,
            ent_coef=0.0001,
            gae_lambda=0.9067,
            n_epochs=3,
            normalize_advantage=True,
            tensorboard_log=directory_logs,
            seed=SEED,
        device='cpu')

    metrics_callback = TrainingMetricsCallback(save_dir=f"./training_results/phase1_{nodes}_nodes", file_name=file)
    
    workload_callback = WorkloadBehaviorCallback(
        total_timesteps=iterations,
        save_dir=f"./training_results/phase1_{nodes}_nodes",
        file_name="workload_behavior.csv"
    )

    model.learn(total_timesteps=iterations, tb_log_name="PPO_Phase1_Simulated", callback=[metrics_callback, workload_callback]) 

    env_sim.save(f"./training_results/phase1_{nodes}_nodes/vec_normalize_phase1.pkl")
    model.save(model_path(nodes))
    print("Fase 1 completada. Conocimiento base guardado.\n")

    print("Generando curva de aprendizaje para Fase 1...")
    viz = Visualizer(save_dir=f"./resultados_graficos/phase1_{nodes}_nodes")
    viz.plot_learning_curve(f"./training_results/phase1_{nodes}_nodes/{file}")
    
    print("Generando gráficos de comportamiento de workload para Fase 1...")
    viz.plot_workload_behavior(
        csv_path=f"./training_results/phase1_{nodes}_nodes/workload_behavior.csv",
        n_max=nodes,
        phase="sim"
    )


def train_phase_2_real_world(nodes=5, iterations=5000, file="training_metrics.csv"):
    print(f"Iniciando entrenamiento con docker + HAProxy para {iterations} pasos con {nodes} nodos...")
    
    raw_env = Monitor(LoadBalancerEnv(simulated=False, n_max=nodes))
    vec_env = DummyVecEnv([lambda: raw_env])
    
    # Cargamos la normalización aprendida en la Fase 1
    phase1_dir = f"./training_results/phase1_{nodes}_nodes"
    if os.path.exists(f"{phase1_dir}/vec_normalize_phase1.pkl"):
        env_real = VecNormalize.load(f"{phase1_dir}/vec_normalize_phase1.pkl", vec_env)
        # La mantenemos entrenando (actualizando promedios) porque la escala real puede variar ligeramente
        env_real.training = True 
        env_real.norm_reward = True
    else:
        env_real = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    if not os.path.exists(f"{model_path(nodes)}.zip"):
        print("No se encontró el modelo base simulado.")
        return

    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PPO.load(model_path(nodes), env=env_real, tensorboard_log=directory_logs, device=device)
    
    model.verbose = 1
    model.lr_schedule = get_schedule_fn(linear_schedule(0.0001))
    model.n_steps = 256 # sobrescribimos n_steps de la fase 1 para generar mas actualizaciones de la policy por falta de recursos y pocas iteraciones
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )

    metrics_callback_real = TrainingMetricsCallback(save_dir=f"./training_results/phase2_{nodes}_nodes", file_name=file)
    logger_callback = StepLoggerCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,                  
        save_path=f'./logs_checkpoints/{nodes}_nodes/',
        name_prefix='ppo_real_env'       
    )
    workload_callback_real = WorkloadBehaviorCallback(
        total_timesteps=iterations,
        save_dir=f"./training_results/phase2_{nodes}_nodes",
        file_name="workload_behavior.csv"
    )


    model.learn(total_timesteps=iterations, tb_log_name="PPO_Phase2_Real_FineTuned", callback=[metrics_callback_real, logger_callback, checkpoint_callback, workload_callback_real])

    env_real.save(f"./training_results/phase2_{nodes}_nodes/vec_normalize_phase2.pkl")
    model.save(f"ppo_lb_production_ready_{nodes}_nodes")
    print("Fase 2 completada.\n")

    print("Generando curva de aprendizaje para Fase 2...")
    viz = Visualizer(save_dir=f"./resultados_graficos/phase2_{nodes}_nodes")
    viz.plot_learning_curve(f"./training_results/phase2_{nodes}_nodes/{file}")
    
    print("Generando gráficos de comportamiento de workload para Fase 2...")
    viz.plot_workload_behavior(
        csv_path=f"./training_results/phase2_{nodes}_nodes/workload_behavior.csv",
        n_max=nodes,
        phase="real"
    )

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
                    learning_rate=linear_schedule(config.learning_rate),
                    gamma=config.gamma,
                    n_steps=config.n_steps,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    clip_range=config.clip_range,
                    ent_coef=config.ent_coef,
                    vf_coef=config.vf_coef,
                    gae_lambda=config.gae_lambda,
                    # target_kl=config.target_kl,
                    normalize_advantage=config.normalize_advantage,
                    tensorboard_log=f"./logs_tensorboard/sweep_{run.id}",
                    seed=SEED,
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
    
    wandb.agent(sweep_id, sweep_train, count=30)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('comando', type=str)
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='training_metrics.csv')
    parser.add_argument('--iterations', type=int, default=5000)
    args = parser.parse_args()

    if args.comando == "train_phase_1_simulation":
        train_phase_1_simulation(nodes=args.nodes, iterations=args.iterations, file=args.file)
    elif args.comando == "train_phase_2_real_world":
        train_phase_2_real_world(nodes=args.nodes, iterations=args.iterations, file=args.file)
    elif args.comando == "sweep":
        run_wandb_sweep(nodes=args.nodes, iterations=args.iterations)
