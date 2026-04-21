import os
import csv
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMetricsCallback(BaseCallback):
    """
    Callback personalizado para guardar el historial de entrenamiento de PPO en un archivo CSV.
    """
    def __init__(self, save_dir: str = "./training_results", verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, "training_metrics.csv")
        os.makedirs(save_dir, exist_ok=True)
        
        # Historial de métricas
        self.rollout_history = {'timestep': [], 'rollout/ep_rew_mean': [], 'rollout/ep_len_mean': []}
        self.train_history = {'train/policy_loss': [], 'train/value_loss': []}
        
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
        except Exception as e: 
            print(f"Error leyendo métrica {key}: {e}")
            pass
            
        return default
    
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        try:
            self.rollout_history['rollout/ep_rew_mean'].append(self._get_logger_value('rollout/ep_rew_mean'))
            self.rollout_history['rollout/ep_len_mean'].append(self._get_logger_value('rollout/ep_len_mean'))
            self.rollout_history['timestep'].append(self.model.num_timesteps)
            
            self.train_history['train/policy_loss'].append(self._get_logger_value('train/policy_gradient_loss'))
            self.train_history['train/value_loss'].append(self._get_logger_value('train/value_loss'))
            
            self._append_to_csv()
        except Exception: pass
