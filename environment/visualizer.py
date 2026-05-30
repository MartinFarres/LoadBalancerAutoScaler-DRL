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

    def _plot_behavior(self, timesteps, workload, active_norm, title, save_path, xlabel='Timestep'):
        """
        Dibuja workload vs contenedores activos para una serie COMPLETA de datos.
        Muestra todos los puntos crudos (tenues) más una media móvil adaptativa
        (línea sólida) para conservar el detalle sin perder la tendencia.
        """
        timesteps = np.asarray(timesteps)
        workload = np.asarray(workload)
        active_norm = np.asarray(active_norm)
        n = len(workload)
        if n == 0:
            return

        fig, ax = plt.subplots(figsize=(14, 5))

        # Datos crudos completos (tenues): no se oculta ningún punto
        ax.plot(timesteps, workload, color='steelblue', alpha=0.25, linewidth=0.7)
        ax.plot(timesteps, active_norm, color='darkorange', alpha=0.25, linewidth=0.7)

        # Media móvil adaptativa al tamaño de la serie para resaltar la tendencia
        window_size = max(1, min(n, max(15, n // 300)))
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            ma_workload = np.convolve(workload, kernel, mode='valid')
            ma_active = np.convolve(active_norm, kernel, mode='valid')
            ma_timesteps = timesteps[window_size - 1:]
        else:
            ma_workload, ma_active, ma_timesteps = workload, active_norm, timesteps

        ax.plot(ma_timesteps, ma_workload, color='steelblue', linewidth=2, label='Workload')
        ax.plot(ma_timesteps, ma_active, color='darkorange', linewidth=2, label='Contenedores Activos')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Valor Normalizado (0 - 1)')
        ax.set_ylim(0, 1.1)
        if timesteps.max() > timesteps.min():
            ax.set_xlim(timesteps.min(), timesteps.max())
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        ax.set_title(title)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Gráfico guardado en: {save_path}")

    def plot_workload_behavior(self, csv_path: str, n_max: int, phase: str = "sim"):
        """
        Genera 3 gráficos (early, middle, late) comparando la carga de trabajo (workload)
        vs la respuesta del agente (contenedores activos normalizados). Cada gráfico
        muestra TODOS los datos de su etapa, no una ventana recortada.
        """
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print("No hay datos para graficar workload behavior.")
                return
        except Exception as e:
            print(f"Error leyendo CSV para workload behavior: {e}")
            return

        df['active_norm'] = df['active_containers'] / n_max

        for stage in ["early", "middle", "late"]:
            subset = df[df['stage'] == stage]
            if subset.empty:
                print(f"No hay datos para la etapa '{stage}' en {csv_path}")
                continue

            self._plot_behavior(
                subset['timestep'].values,
                subset['workload'].values,
                subset['active_norm'].values,
                title=f"Comportamiento del Agente vs Workload — {stage.capitalize()} ({phase.upper()})",
                save_path=os.path.join(self.save_dir, f"workload_behavior_{phase}_{stage}.png"),
                xlabel='Timestep',
            )

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
        Genera gráficos comparando workload vs contenedores activos a partir del CSV
        de testing. Muestra TODOS los datos: un gráfico con la corrida completa y,
        además, uno por etapa (early, middle, late) con el detalle de cada tercio.
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
        suffix = f"_{mode}" if mode else ""

        # Corrida COMPLETA: todos los datos en un solo gráfico
        self._plot_behavior(
            df['step'].values,
            df['workload'].values,
            df['active_norm'].values,
            title=f"Comportamiento vs Workload — Corrida Completa ({mode.upper()})",
            save_path=os.path.join(self.save_dir, f"workload_behavior{suffix}_full.png"),
            xlabel='Step',
        )

        # Detalle por etapa (cada tercio de la corrida, con todos sus datos)
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

            self._plot_behavior(
                subset['step'].values,
                subset['workload'].values,
                subset['active_norm'].values,
                title=f"Comportamiento vs Workload — {stage.capitalize()} ({mode.upper()})",
                save_path=os.path.join(self.save_dir, f"workload_behavior{suffix}_{stage}.png"),
                xlabel='Step',
            )