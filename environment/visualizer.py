import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, save_dir="./resultados_graficos"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_learning_curve(self, csv_path):
        """
        Genera la Curva de Aprendizaje leyendo el CSV de training_metrics.
        Aplica una media móvil para suavizar el ruido típico de RL.
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Filtramos filas donde haya reward registrado
            df_clean = df[df['rollout/ep_rew_mean'].notna() & (df['rollout/ep_rew_mean'] != 0.0)]
            
            if df_clean.empty:
                print("No hay suficientes datos de reward para graficar.")
                return

            timesteps = df_clean['timestep'].values
            rewards = df_clean['rollout/ep_rew_mean'].values

            fig, ax = plt.subplots(figsize=(10, 6))

            # Gráfico de fondo (datos crudos) transparentes
            ax.plot(timesteps, rewards, color='steelblue', alpha=0.3, label='Reward Crudo')

            # Calcular Media Móvil (Window = 20)
            window = min(20, max(1, len(rewards) // 5))
            if len(rewards) >= window:
                weights = np.ones(window) / window
                ma_rewards = np.convolve(rewards, weights, mode='valid')
                ma_timesteps = timesteps[window - 1:]
                ax.plot(ma_timesteps, ma_rewards, color='darkblue', linewidth=2, label=f'Media Móvil (n={window})')

            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Reward Promedio por Episodio')
            ax.set_title('Curva de Aprendizaje (PPO)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

            save_path = os.path.join(self.save_dir, 'learning_curve.png')
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Curva de aprendizaje guardada en: {save_path}")

        except Exception as e:
            print(f"Error generando curva de aprendizaje: {e}")

    def generate_testing_summary_table(self, cpu_history, ram_history, latency_history, errors_history, scaling_events=0, sla_violation_pct=0.0, avg_cost_efficiency=0.0, high_latency_threshold=0.8, mode=''):
        """
        Genera una tabla resumen guardada como imagen con las métricas 
        """
        total_failed_requests = sum(errors_history)
        avg_cpu = np.mean(cpu_history) * 100 
        avg_ram = np.mean(ram_history) * 100 

        # Preparar datos
        cell_text = [
            [f"{int(total_failed_requests)}"],
            [f"{avg_cpu:.2f}%"],
            [f"{avg_ram:.2f}%"],
            [f"{scaling_events}"],                    
            [f"{sla_violation_pct:.2f}%"],            
            [f"{avg_cost_efficiency:.1f} usr/nodo"]   
        ]
        
        rows = [
            'Failed Requests (Total)', 
            'Average CPU Usage', 
            'Average Memory Usage', 
            'Scaling Events (Chattering)', 
            'SLA Violations (>1000ms)', 
            'Cost Efficiency'
        ]

        # Se aumenta de 3 a 4.5 el alto de la figura para que entren las nuevas filas cómodamente
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=['Valor Obtenido'], loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        suffix = f"_{mode}" if mode else ""
        save_path = os.path.join(self.save_dir, f'testing_summary_table{suffix}.png')
        plt.title('Resumen de Métricas (Fase de Pruebas)', fontweight="bold", pad=20)
        fig.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Tabla de resumen guardada en: {save_path}")

    def plot_workload_behavior(self, csv_path: str, n_max: int, phase: str = "sim"):
        """
        Genera 3 gráficos independientes (uno por etapa: early, middle, late)
        comparando la carga de trabajo (workload) vs la respuesta del agente
        (contenedores activos normalizados).
        """
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print("No hay datos para graficar workload behavior.")
                return
        except Exception as e:
            print(f"Error leyendo CSV para workload behavior: {e}")
            return

        # Paso 2: Normalizar active_containers a escala 0-1
        df['active_norm'] = df['active_containers'] / n_max

        # Paso 3: Iterar sobre las tres etapas
        for stage in ["early", "middle", "late"]:
            subset = df[df['stage'] == stage]
            if subset.empty:
                print(f"No hay datos para la etapa '{stage}' en {csv_path}")
                continue

            # Paso 4: Seleccionar ventana representativa de 600 steps
            center = len(subset) // 2
            start = max(0, center - 300)
            end = min(len(subset), center + 300)
            window = subset.iloc[start:end]

            if window.empty:
                continue

            timesteps = window['timestep'].values
            workload = window['workload'].values
            active_norm = window['active_norm'].values

            # Paso 5: Aplicar media móvil (ventana 15)
            window_size = 15
            if len(workload) >= window_size:
                ma_workload = np.convolve(workload, np.ones(window_size)/window_size, mode='valid')
                ma_active = np.convolve(active_norm, np.ones(window_size)/window_size, mode='valid')
                ma_timesteps = timesteps[window_size-1:]
            else:
                ma_workload = workload
                ma_active = active_norm
                ma_timesteps = timesteps

            # Paso 6: Generar la figura
            fig, ax = plt.subplots(figsize=(12, 5))

            # Medias móviles (solo se grafican estas, sin datos crudos)
            ax.plot(ma_timesteps, ma_workload, color='steelblue', linewidth=2, label='Workload')
            ax.plot(ma_timesteps, ma_active, color='darkorange', linewidth=2, label='Contenedores Activos')

            # Configuración
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Valor Normalizado (0 - 1)')
            ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')
            ax.set_title(f"Comportamiento del Agente vs Workload — {stage.capitalize()} ({phase.upper()})")

            # Paso 7: Guardar
            save_path = os.path.join(self.save_dir, f"workload_behavior_{phase}_{stage}.png")
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Gráfico guardado en: {save_path}")

    def plot_testing_reward_curve(self, csv_path: str, mode: str = ''):
        """
        Genera la curva de recompensa durante la fase de pruebas.
        Equivalente a plot_learning_curve pero para el CSV de testing.
        """
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print("No hay datos para graficar la curva de recompensa.")
                return

            steps = df['step'].values
            rewards = df['reward'].values

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(steps, rewards, color='steelblue', alpha=0.3, label='Reward Crudo')

            window = min(20, max(1, len(rewards) // 5))
            if len(rewards) >= window:
                weights = np.ones(window) / window
                ma_rewards = np.convolve(rewards, weights, mode='valid')
                ma_steps = steps[window - 1:]
                ax.plot(ma_steps, ma_rewards, color='darkblue', linewidth=2, label=f'Media Móvil (n={window})')

            ax.set_xlabel('Step')
            ax.set_ylabel('Recompensa por Step')
            ax.set_title(f'Curva de Recompensa — Fase de Pruebas ({mode.upper()})')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

            suffix = f"_{mode}" if mode else ""
            save_path = os.path.join(self.save_dir, f'testing_reward_curve{suffix}.png')
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Curva de recompensa guardada en: {save_path}")

        except Exception as e:
            print(f"Error generando curva de recompensa: {e}")

    def plot_testing_behavior(self, csv_path: str, n_max: int, mode: str = ''):
        """
        Genera 3 gráficos independientes (early, middle, late) comparando workload
        vs contenedores activos normalizados, a partir del CSV de testing.
        Equivalente a plot_workload_behavior pero lee el formato del CSV de testing.
        """
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print("No hay datos para graficar testing behavior.")
                return
        except Exception as e:
            print(f"Error leyendo CSV para testing behavior: {e}")
            return

        df['active_norm'] = df['activos'] / n_max

        total = len(df)
        third = max(1, total // 3)

        stages = {
            "early":  df.iloc[:third],
            "middle": df.iloc[third:2 * third],
            "late":   df.iloc[2 * third:],
        }

        for stage, subset in stages.items():
            if subset.empty:
                print(f"No hay datos para la etapa '{stage}'.")
                continue

            center = len(subset) // 2
            start = max(0, center - 300)
            end = min(len(subset), center + 300)
            window = subset.iloc[start:end]

            if window.empty:
                continue

            timesteps = window['step'].values
            workload = window['workload'].values
            active_norm = window['active_norm'].values

            window_size = 15
            if len(workload) >= window_size:
                ma_workload = np.convolve(workload, np.ones(window_size) / window_size, mode='valid')
                ma_active = np.convolve(active_norm, np.ones(window_size) / window_size, mode='valid')
                ma_timesteps = timesteps[window_size - 1:]
            else:
                ma_workload = workload
                ma_active = active_norm
                ma_timesteps = timesteps

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ma_timesteps, ma_workload, color='steelblue', linewidth=2, label='Workload')
            ax.plot(ma_timesteps, ma_active, color='darkorange', linewidth=2, label='Contenedores Activos')

            ax.set_xlabel('Step')
            ax.set_ylabel('Valor Normalizado (0 - 1)')
            ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')
            ax.set_title(f"Comportamiento vs Workload — {stage.capitalize()} ({mode.upper()})")

            suffix = f"_{mode}" if mode else ""
            save_path = os.path.join(self.save_dir, f"workload_behavior{suffix}_{stage}.png")
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Gráfico guardado en: {save_path}")