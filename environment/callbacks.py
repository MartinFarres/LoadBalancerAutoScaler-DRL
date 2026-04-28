import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMetricsCallback(BaseCallback):
    """
    Callback personalizado para guardar el historial de entrenamiento de PPO en un archivo CSV.
    """
    def __init__(self, save_dir: str = "./training_results", file_name: str = "training_metrics.csv", verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)
        
        self.rollout_history = {
            'timestep': [], 
            'rollout/ep_rew_mean': [], 
            'rollout/ep_len_mean': [],
            'rollout/cpu_mean': [],       
            'rollout/ram_mean': [],
            'rollout/latency_mean': [],
            'rollout/error_mean': []
        }
        self.train_history = {'train/policy_loss': [], 'train/value_loss': []}
        
        self.step_metrics_buffer = {'cpu': [], 'ram': [], 'latency': [], 'errors': []}
        
        self._init_csv()
    
    def _init_csv(self):
        all_keys = list(self.rollout_history.keys()) + list(self.train_history.keys())
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
    
    def _append_to_csv(self):
        if len(self.rollout_history['timestep']) == 0: return
        row = {}
        for key, values in self.rollout_history.items():
            row[key] = values[-1] if len(values) > 0 else 0.0
        for key, values in self.train_history.items():
            row[key] = values[-1] if len(values) > 0 else 0.0
            
        all_keys = list(self.rollout_history.keys()) + list(self.train_history.keys())
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writerow(row)
    
    def _get_logger_value(self, key: str, default: float = 0.0):
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                if key in self.model.logger.name_to_value:
                    return float(self.model.logger.name_to_value[key])
        except Exception: pass
        return default

    def _get_ep_info(self, key: str, default: float = 0.0):
        try:
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                if key == 'rollout/ep_rew_mean':
                    return float(np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]))
                elif key == 'rollout/ep_len_mean':
                    return float(np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]))
        except Exception: pass
        return default
    
    def _on_step(self):
        # Capturamos las métricas del diccionario 'info' del entorno
        for info in self.locals.get("infos", []):
            if "cpu_avg" in info:
                self.step_metrics_buffer['cpu'].append(info['cpu_avg'])
                self.step_metrics_buffer['ram'].append(info['ram_avg'])
                self.step_metrics_buffer['latency'].append(info['latency_avg'])
                self.step_metrics_buffer['errors'].append(info['error_avg'])
        return True
    
    def _on_rollout_end(self):
        try:
            # Métricas estándar
            self.rollout_history['rollout/ep_rew_mean'].append(self._get_ep_info('rollout/ep_rew_mean'))
            self.rollout_history['rollout/ep_len_mean'].append(self._get_ep_info('rollout/ep_len_mean'))
            self.rollout_history['timestep'].append(self.model.num_timesteps)
            
            # Promediamos y guardamos las métricas de hardware del último rollout
            self.rollout_history['rollout/cpu_mean'].append(
                np.mean(self.step_metrics_buffer['cpu']) if self.step_metrics_buffer['cpu'] else 0.0)
            self.rollout_history['rollout/ram_mean'].append(
                np.mean(self.step_metrics_buffer['ram']) if self.step_metrics_buffer['ram'] else 0.0)
            self.rollout_history['rollout/latency_mean'].append(
                np.mean(self.step_metrics_buffer['latency']) if self.step_metrics_buffer['latency'] else 0.0)
            self.rollout_history['rollout/error_mean'].append(
                np.mean(self.step_metrics_buffer['errors']) if self.step_metrics_buffer['errors'] else 0.0)
            
            # Limpiamos el buffer para el próximo ciclo
            self.step_metrics_buffer = {'cpu': [], 'ram': [], 'latency': [], 'errors': []}
            
            # Métricas de pérdida
            self.train_history['train/policy_loss'].append(self._get_logger_value('train/policy_gradient_loss'))
            self.train_history['train/value_loss'].append(self._get_logger_value('train/value_loss'))
            
            self._append_to_csv()
        except Exception: 
            pass