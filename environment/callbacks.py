# TrainingMetricsCallback - El que hace de "cerebro" del seguimiento de entrenamiento PPO
# =======================================================================================
# Este callback esta constantemente vigilando como evoluciona el entrenamiento, como un
# entrenador personal que anota cada detalle relevante:
#
# 1. Como le va al modelo (rewards, losses, entropia)
# 2. Si los gradientes se estan comportando bien o si hay problemas (normas L2)
#
# Uso:
#     from callbacks import TrainingMetricsCallback
#     callback = TrainingMetricsCallback(save_dir="./training_results")
#     model.learn(total_timesteps=10000, callback=callback)

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    # El callback que registra todo lo que pasa durante el entrenamiento.
    # Va anotando:
    #     - Metricas de rollout: reward promedio que obtiene el agente, cuanto duran los episodios
    #     - Metricas de entrenamiento: policy_loss (que tan mal le esta errando al agente),
    #       value_loss (que tan mal predice valores), entropy (cuanto esta explorando),
    #       clip_fraction (que tanto esta limitado por el clipping)
    #     - Metricas de gradiente: norma L2 de gradientes de policy y value networks
    # Al final genera:
    #     - Un CSV con todas las metricas para poder analizar todo en detalle
    
    def __init__(self, save_dir: str = "./training_results", verbose: int = 0):
        
        super().__init__(verbose) # 1 = muestra info por pantalla, 0 = silencioso
        
        self.save_dir = save_dir # Donde guardamos todos los resultados del entrenamiento
        self.csv_path = os.path.join(save_dir, "training_metrics.csv")
        
        # Si el directorio no existe, lo creamos
        os.makedirs(save_dir, exist_ok=True)
        
        # Aqui guardamos el historial de metricas de rollout (datos del entorno)
        self.rollout_history = {}
        self.rollout_history['timestep'] = []
        self.rollout_history['rollout/ep_rew_mean'] = []
        self.rollout_history['rollout/ep_len_mean'] = []
        
        # Metricas de training (se capturan despues de cada actualizacion del modelo)
        self.train_history = {}
        self.train_history['train/policy_loss'] = []
        self.train_history['train/value_loss'] = []
        self.train_history['train/entropy_loss'] = []
        self.train_history['train/clip_fraction'] = []
        self.train_history['train/explained_variance'] = []
        
        # Metricas de gradiente (se capturan periodicamente durante las actualizaciones)
        self.gradient_history = {}
        self.gradient_history['timestep'] = []
        self.gradient_history['gradient/policy_norm'] = []
        self.gradient_history['gradient/value_norm'] = []
        
        # Creamos el archivo CSV y escribimos los headers (nombres de columnas)
        self._init_csv()
        
        if self.verbose > 0:
            print("[MetricsCallback] Guardando metricas en: " + self.save_dir)
    
    def _init_csv(self):
        # Abre el archivo CSV y escribe los headers (nombres de cada columna).
        all_keys = []
        all_keys.extend(self.rollout_history.keys())
        all_keys.extend(self.train_history.keys())
        all_keys.extend(self.gradient_history.keys())
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
    
    def _append_to_csv(self):
        # Toma los ultimos valores de todas las metricas y los agrega como una fila nueva al CSV.
        if len(self.rollout_history['timestep']) == 0:
            return
        
        # Armamos un diccionario con todos los valores para esta fila
        row = {}
        
        for key, values in self.rollout_history.items():
            if len(values) > 0:
                row[key] = values[-1]
            else:
                row[key] = 0.0
                
        for key, values in self.train_history.items():
            if len(values) > 0:
                row[key] = values[-1]
            else:
                row[key] = 0.0
                
        for key, values in self.gradient_history.items():
            if len(values) > 0:
                row[key] = values[-1]
            else:
                row[key] = 0.0
        
        all_keys = []
        all_keys.extend(self.rollout_history.keys())
        all_keys.extend(self.train_history.keys())
        all_keys.extend(self.gradient_history.keys())
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writerow(row)
    
    def _get_logger_value(self, key: str, default: float = 0.0):
        # Busca una metrica especifica en los loggers del modelo y entorno.
        
        value = None
        
        # Primero vemos si esta en el logger del entorno (el Monitor)
        # Ahi se guardan las metricas de rollout como ep_rew_mean y ep_len_mean
        try:
            if hasattr(self.model, 'env'):
                env = self.model.env
                # El Monitor puede estar envuelto dentro de otros wrappers, hay que buscar bien
                while hasattr(env, 'env'):
                    if hasattr(env, 'logger') and hasattr(env.logger, 'name_to_value'):
                        value = env.logger.name_to_value.get(key, None)
                        if value is not None:
                            return float(value)
                    env = env.env
                
                # Por las dudas, tambien verificamos el env directamente
                if hasattr(env, 'logger') and hasattr(env.logger, 'name_to_value'):
                    value = env.logger.name_to_value.get(key, None)
                    if value is not None:
                        return float(value)
        except Exception:
            pass
        
        # Si no encontramos la metrica ahi, buscamos en el logger del modelo PPO
        try:
            if hasattr(self.model, 'logger'):
                logger = self.model.logger
                
                # Primero vemos en name_to_value (diccionario directo)
                if hasattr(logger, 'name_to_value'):
                    value = logger.name_to_value.get(key, None)
                    if value is not None:
                        return float(value)
                
                # Y si no, buscamos en output_formats (formato tensorboard)
                if hasattr(logger, 'output_formats'):
                    for fmt in logger.output_formats:
                        if hasattr(fmt, 'name_to_value') and key in fmt.name_to_value:
                            value = fmt.name_to_value[key]
                            return float(value)
        except Exception:
            pass
        
        return default
    
    def _on_step(self):
        # Se ejecuta automaticamente despues de cada step del entorno.
        # Aqui capturamos metricas de gradiente de vez en cuando.
        # Capturamos los gradientes cada 512 steps (mas o menos al final de cada rollout)
        current_step = self.model.num_timesteps
        if current_step % 512 == 0 and current_step > 0:
            self._capture_gradients()
        
        return True
    
    def _capture_gradients(self):
        # Calcula y guarda la norma L2 de los gradientes de la red.
        try:
            policy_norm = 0.0
            value_norm = 0.0
            
            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    # Convertimos el gradiente a numpy para poder calcular su norma
                    grad_data = param.grad.detach().cpu().numpy()
                    grad_norm = float(np.linalg.norm(grad_data))
                    
                    # Clasificamos si pertenece a la red de policy (actor) o value (critic)
                    name_lower = name.lower()
                    if 'actor' in name_lower or 'policy' in name_lower:
                        policy_norm = policy_norm + (grad_norm ** 2)
                    else:
                        value_norm = value_norm + (grad_norm ** 2)
            
            # Guardamos las normas (la norma L2 es la raiz cuadrada de la suma de cuadrados)
            current_timestep = self.model.num_timesteps
            self.gradient_history['timestep'].append(current_timestep)
            self.gradient_history['gradient/policy_norm'].append(float(np.sqrt(policy_norm)))
            self.gradient_history['gradient/value_norm'].append(float(np.sqrt(value_norm)))
            
        except Exception as e:
            if self.verbose > 0:
                print("[MetricsCallback] Error calculando gradientes: " + str(e))
    
    def _on_rollout_end(self):
        # Se llama automaticamente cuando termina un rollout.
        # Aca registramos metricas de recompensa y episodios.
        try:
            # Obtenemos las metricas de rollout del Monitor
            ep_rew = self._get_logger_value('rollout/ep_rew_mean', 0.0)
            ep_len = self._get_logger_value('rollout/ep_len_mean', 0.0)
            
            self.rollout_history['rollout/ep_rew_mean'].append(ep_rew)
            self.rollout_history['rollout/ep_len_mean'].append(ep_len)
            self.rollout_history['timestep'].append(self.model.num_timesteps)
            
            # Ahora obtenemos las metricas de entrenamiento (estan disponibles despues de actualizar el modelo)
            policy_loss = self._get_logger_value('train/policy_gradient_loss', 0.0)
            value_loss = self._get_logger_value('train/value_loss', 0.0)
            entropy_loss = self._get_logger_value('train/entropy_loss', 0.0)
            clip_fraction = self._get_logger_value('train/clip_fraction', 0.0)
            explained_variance = self._get_logger_value('train/explained_variance', 0.0)
            
            self.train_history['train/policy_loss'].append(policy_loss)
            self.train_history['train/value_loss'].append(value_loss)
            self.train_history['train/entropy_loss'].append(entropy_loss)
            self.train_history['train/clip_fraction'].append(clip_fraction)
            self.train_history['train/explained_variance'].append(explained_variance)
            
            # Y guardamos todo en el CSV para tener registro
            self._append_to_csv()
            
        except Exception as e:
            if self.verbose > 0:
                print("[MetricsCallback] Warning en rollout: " + str(e))
        
        return None
    
    def _on_training_end(self):
        # Se llama automaticamente cuando termina todo el entrenamiento.
        # Aca generamos los graficos de resumen para poder visualizar como fue todo.
        if self.verbose > 0:
            print("[MetricsCallback] Entrenamiento finalizado. Generando graficos...")
        
        self._generate_plots()
        
        if self.verbose > 0:
            print("[MetricsCallback] Metricas guardadas en: " + self.csv_path)
        
        return None
    
    def _generate_plots(self):
        # Acá generamos todos los gráficos de resumen del entrenamiento.
        # Usamos barras (bar charts) para que sea fácil de leer.
        # Si no tenemos suficientes datos, no tiene sentido hacer gráficos
        if len(self.rollout_history['timestep']) < 2:
            print("[MetricsCallback] No hay suficientes datos para generar graficos")
            return
        
        # Convertimos todo a arrays de numpy para poder graficar
        timesteps = np.array(self.rollout_history['timestep'])
        rewards = np.array(self.rollout_history['rollout/ep_rew_mean'])
        ep_lens = np.array(self.rollout_history['rollout/ep_len_mean'])
        
        policy_loss = np.array(self.train_history['train/policy_loss'])
        value_loss = np.array(self.train_history['train/value_loss'])
        entropy = np.array(self.train_history['train/entropy_loss'])
        clip_frac = np.array(self.train_history['train/clip_fraction'])
        explained_var = np.array(self.train_history['train/explained_variance'])
        
        # Calculamos a que timesteps corresponden las metricas de training
        train_timesteps = timesteps[:len(policy_loss)]
        
        # Para los graficos de barras, si hay muchos datos solo mostramos algunos puntos
        # (si no las barras quedan muy juntas y no se ve nada)
        max_bars = 100
        show_every = max(1, len(timesteps) // max_bars)
        
        # Seleccionamos indices para muestrear (cada show_every puntos)
        indices = list(range(0, len(timesteps), show_every))
        if len(timesteps) - 1 not in indices:
            indices.append(len(timesteps) - 1)
        
        # ============================================================
        # GRAFICO 1: Como fue evolucionando el Reward (Barras)
        # ============================================================
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        x_pos = np.arange(len(timesteps))
        ax1.bar(x_pos, rewards, color='steelblue', alpha=0.7, label='Reward Promedio')
        
        # Ponemos etiquetas solo en algunos puntos para que no se superpongan
        if len(timesteps) > max_bars:
            ax1.set_xticks(indices)
            ax1.set_xticklabels([str(timesteps[i]) for i in indices], rotation=45)
        else:
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([str(int(t)) for t in timesteps], rotation=45)
        
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Reward (acumulado)')
        ax1.set_title('Evolucion del Reward durante Entrenamiento')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.save_dir, 'plot_1_reward_evolution.png'), dpi=150)
        plt.close(fig1)
        
        # ============================================================
        # GRAFICO 2: Como evolucionaron las Losses (Policy y Value) - Barras
        # ============================================================
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        
        # Policy Loss
        x_pos_policy = np.arange(len(policy_loss))
        axes2[0].bar(x_pos_policy, policy_loss, color='green', alpha=0.7)
        if len(policy_loss) > max_bars:
            policy_indices = list(range(0, len(policy_loss), show_every))
            if len(policy_loss) - 1 not in policy_indices:
                policy_indices.append(len(policy_loss) - 1)
            axes2[0].set_xticks(policy_indices)
            axes2[0].set_xticklabels([str(train_timesteps[i]) for i in policy_indices], rotation=45)
        else:
            axes2[0].set_xticks(x_pos_policy)
            axes2[0].set_xticklabels([str(int(t)) for t in train_timesteps], rotation=45)
        axes2[0].set_xlabel('Timesteps')
        axes2[0].set_ylabel('Policy Loss (negativo)')
        axes2[0].set_title('Evolucion del Policy Loss')
        axes2[0].grid(True, alpha=0.3, axis='y')
        
        # Value Loss
        x_pos_value = np.arange(len(value_loss))
        axes2[1].bar(x_pos_value, value_loss, color='purple', alpha=0.7)
        if len(value_loss) > max_bars:
            value_indices = list(range(0, len(value_loss), show_every))
            if len(value_loss) - 1 not in value_indices:
                value_indices.append(len(value_loss) - 1)
            axes2[1].set_xticks(value_indices)
            axes2[1].set_xticklabels([str(train_timesteps[i]) for i in value_indices], rotation=45)
        else:
            axes2[1].set_xticks(x_pos_value)
            axes2[1].set_xticklabels([str(int(t)) for t in train_timesteps], rotation=45)
        axes2[1].set_xlabel('Timesteps')
        axes2[1].set_ylabel('Value Loss (MSE)')
        axes2[1].set_title('Evolucion del Value Loss')
        axes2[1].grid(True, alpha=0.3, axis='y')
        
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.save_dir, 'plot_2_losses.png'), dpi=150)
        plt.close(fig2)
        
        # ============================================================
        # GRAFICO 3: Como变化 la Entropia (que tanto explora el agente) - Barras
        # ============================================================
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        
        x_pos_entropy = np.arange(len(entropy))
        ax3.bar(x_pos_entropy, entropy, color='cyan', alpha=0.7)
        if len(entropy) > max_bars:
            entropy_indices = list(range(0, len(entropy), show_every))
            if len(entropy) - 1 not in entropy_indices:
                entropy_indices.append(len(entropy) - 1)
            ax3.set_xticks(entropy_indices)
            ax3.set_xticklabels([str(train_timesteps[i]) for i in entropy_indices], rotation=45)
        else:
            ax3.set_xticks(x_pos_entropy)
            ax3.set_xticklabels([str(int(t)) for t in train_timesteps], rotation=45)
        ax3.set_xlabel('Timesteps')
        ax3.set_ylabel('Entropy (nats)')
        ax3.set_title('Entropia (Exploracion del Agente)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        fig3.tight_layout()
        fig3.savefig(os.path.join(self.save_dir, 'plot_3_entropy.png'), dpi=150)
        plt.close(fig3)
        
        # ============================================================
        # GRAFICO 4: Norma de Gradientes (estabilidad del entrenamiento) - Barras
        # ============================================================
        if len(self.gradient_history['timestep']) > 0:
            grad_norms = np.array(self.gradient_history['gradient/policy_norm'])
            grad_timesteps = np.array(self.gradient_history['timestep'])
            
            if np.any(grad_norms > 0):
                fig4, ax4 = plt.subplots(figsize=(12, 5))
                
                x_pos_grad = np.arange(len(grad_norms))
                ax4.bar(x_pos_grad, grad_norms, color='purple', alpha=0.7)
                
                # Ponemos las etiquetas en el eje X
                if len(grad_norms) > max_bars:
                    grad_indices = list(range(0, len(grad_norms), show_every))
                    if len(grad_norms) - 1 not in grad_indices:
                        grad_indices.append(len(grad_norms) - 1)
                    ax4.set_xticks(grad_indices)
                    ax4.set_xticklabels([str(int(grad_timesteps[i])) for i in grad_indices], rotation=45)
                else:
                    ax4.set_xticks(x_pos_grad)
                    ax4.set_xticklabels([str(int(t)) for t in grad_timesteps], rotation=45)
                
                ax4.set_xlabel('Timesteps')
                ax4.set_ylabel('Grad Norm L2')
                ax4.set_title('Estabilidad del Gradiente (Policy Network)')
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Detectamos si hubo gradient explosion
                max_grad = np.max(grad_norms)
                if max_grad > 100:
                    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7)
                    ax4.text(0, 100, ' Posible Gradient Explosion', fontsize=10, color='red')
                
                fig4.tight_layout()
                fig4.savefig(os.path.join(self.save_dir, 'plot_4_gradient_norms.png'), dpi=150)
                plt.close(fig4)
        
        # ============================================================
        # GRAFICO 5: Vista general de todo - Barras
        # ============================================================
        fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
        fig5.suptitle('Resumen de Entrenamiento PPO', fontsize=14, fontweight='bold')
        
        # Reward
        x_pos_r = np.arange(len(timesteps))
        axes5[0, 0].bar(x_pos_r, rewards, color='steelblue', alpha=0.7)
        axes5[0, 0].set_title('Reward (acumulado)')
        axes5[0, 0].set_xlabel('Timesteps')
        axes5[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Policy Loss
        x_pos_pl = np.arange(len(policy_loss))
        axes5[0, 1].bar(x_pos_pl, policy_loss, color='green', alpha=0.7)
        axes5[0, 1].set_title('Policy Loss (negativo)')
        axes5[0, 1].set_xlabel('Timesteps')
        axes5[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Value Loss
        x_pos_vl = np.arange(len(value_loss))
        axes5[0, 2].bar(x_pos_vl, value_loss, color='purple', alpha=0.7)
        axes5[0, 2].set_title('Value Loss (MSE)')
        axes5[0, 2].set_xlabel('Timesteps')
        axes5[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Entropy
        x_pos_en = np.arange(len(entropy))
        axes5[1, 0].bar(x_pos_en, entropy, color='cyan', alpha=0.7)
        axes5[1, 0].set_title('Entropy (nats)')
        axes5[1, 0].set_xlabel('Timesteps')
        axes5[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Clip Fraction
        x_pos_cf = np.arange(len(clip_frac))
        axes5[1, 1].bar(x_pos_cf, clip_frac, color='magenta', alpha=0.7)
        axes5[1, 1].axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
        axes5[1, 1].set_title('Clip Fraction (0-1)')
        axes5[1, 1].set_xlabel('Timesteps')
        axes5[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Episode Length
        x_pos_el = np.arange(len(ep_lens))
        axes5[1, 2].bar(x_pos_el, ep_lens, color='brown', alpha=0.7)
        axes5[1, 2].set_title('Longitud Episodios (steps)')
        axes5[1, 2].set_xlabel('Timesteps')
        axes5[1, 2].grid(True, alpha=0.3, axis='y')
        
        fig5.tight_layout()
        fig5.savefig(os.path.join(self.save_dir, 'plot_5_summary.png'), dpi=150)
        plt.close(fig5)
        
        # ============================================================
        # GRAFICO 6: Metricas Adicionales (Clip Fraction y Explained Variance) - Barras
        # ============================================================
        fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
        
        # Clip Fraction
        x_pos_cf = np.arange(len(clip_frac))
        axes6[0].bar(x_pos_cf, clip_frac, color='magenta', alpha=0.7)
        axes6[0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Umbral 0.2')
        if len(clip_frac) > max_bars:
            cf_indices = list(range(0, len(clip_frac), show_every))
            if len(clip_frac) - 1 not in cf_indices:
                cf_indices.append(len(clip_frac) - 1)
            axes6[0].set_xticks(cf_indices)
            axes6[0].set_xticklabels([str(train_timesteps[i]) for i in cf_indices], rotation=45)
        else:
            axes6[0].set_xticks(x_pos_cf)
            axes6[0].set_xticklabels([str(int(t)) for t in train_timesteps], rotation=45)
        axes6[0].set_xlabel('Timesteps')
        axes6[0].set_ylabel('Clip Fraction (0-1)')
        axes6[0].set_title('Fraccion de Updates Recortados')
        axes6[0].legend()
        axes6[0].grid(True, alpha=0.3, axis='y')
        
        # Explained Variance
        x_pos_ev = np.arange(len(explained_var))
        axes6[1].bar(x_pos_ev, explained_var, color='navy', alpha=0.7)
        axes6[1].axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label='Linea Cero')
        if len(explained_var) > max_bars:
            ev_indices = list(range(0, len(explained_var), show_every))
            if len(explained_var) - 1 not in ev_indices:
                ev_indices.append(len(explained_var) - 1)
            axes6[1].set_xticks(ev_indices)
            axes6[1].set_xticklabels([str(train_timesteps[i]) for i in ev_indices], rotation=45)
        else:
            axes6[1].set_xticks(x_pos_ev)
            axes6[1].set_xticklabels([str(int(t)) for t in train_timesteps], rotation=45)
        axes6[1].set_xlabel('Timesteps')
        axes6[1].set_ylabel('Explained Variance (0-1)')
        axes6[1].set_title('Varianza Explicada por Value Function')
        axes6[1].legend()
        axes6[1].grid(True, alpha=0.3, axis='y')
        
        fig6.tight_layout()
        fig6.savefig(os.path.join(self.save_dir, 'plot_6_additional_metrics.png'), dpi=150)
        plt.close(fig6)
        
        print("[MetricsCallback] Graficos guardados en: " + self.save_dir)


class GradientCallback(BaseCallback):
    # Callback enfocado solo en monitorear los gradientes.
    # Sirve para detectar:
    #     - Gradient Explosion: cuando la norma supera 100 (los pesos se disparan)
    #     - Vanishing Gradients: cuando la norma es ~ 0 (los pesos apenas cambian)
    #     - Inestabilidad en el entrenamiento
    # Uso:
    #     gradient_cb = GradientCallback(check_freq=100)
    #     model.learn(..., callback=gradient_cb)
    
    def __init__(self, check_freq: int = 100, verbose: int = 0):
        # Args:
        #     check_freq: Cada cuanto sampledamos los gradientes (en timesteps)
        #     verbose: Si es 1, muestra advertencias por pantalla
        super().__init__(verbose)
        self.check_freq = check_freq
        self.gradient_history = []
        self.timestep_history = []
        
    def _on_step(self) -> bool:
        # Verifica los gradientes cada check_freq timesteps.
        current_step = self.model.num_timesteps
        
        if current_step % self.check_freq == 0:
            policy_grad_norm = 0.0
            
            for param in self.model.policy.parameters():
                if param.grad is not None:
                    grad_np = param.grad.cpu().numpy()
                    policy_grad_norm = policy_grad_norm + np.sum(grad_np ** 2)
            
            policy_grad_norm = np.sqrt(policy_grad_norm)
            
            self.gradient_history.append(policy_grad_norm)
            self.timestep_history.append(current_step)
            
            # Y si esta todo en modo verbose, advertimos si hay anomalias
            if self.verbose > 0:
                if policy_grad_norm > 100:
                    print("[GradientCallback] Gradient Explosion: " + str(policy_grad_norm))
                elif policy_grad_norm < 0.001:
                    print("[GradientCallback] Vanishing Gradient: " + str(policy_grad_norm))
        
        return True
    
    def plot_gradients(self, save_path: str = "gradient_evolution.png"):
        # Genera un grafico de barras mostrando como evolucionaron los gradientes.
        if len(self.gradient_history) == 0:
            print("[GradientCallback] No hay datos de gradientes")
            return
        
        plt.figure(figsize=(10, 5))
        
        x_pos = np.arange(len(self.gradient_history))
        plt.bar(x_pos, self.gradient_history, color='purple', alpha=0.7)
        
        plt.axhline(y=100, color='red', linestyle='--', label='Umbral Explosion (100)')
        plt.xlabel('Timesteps')
        plt.ylabel('Norma L2 del Gradiente')
        plt.title('Evolucion de Gradientes durante Entrenamiento')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print("[GradientCallback] Grafico guardado: " + save_path)
