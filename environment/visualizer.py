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

    def generate_testing_summary_table(self, cpu_history, ram_history, latency_history, errors_history, high_latency_threshold=0.8):
        """
        Genera una tabla resumen guardada como imagen con las métricas 
        """
        # metricas
        total_failed_requests = sum(errors_history)
        avg_cpu = np.mean(cpu_history) * 100  # Convertido a %
        avg_ram = np.mean(ram_history) * 100  # Convertido a %
        
        # Eventos de alta latencia  ( latencia > 80% )
        high_latency_events = sum(1 for lat in latency_history if lat >= high_latency_threshold)

        # Preparar datos
        cell_text = [
            [f"{int(total_failed_requests)}"],
            [f"{avg_cpu:.2f}%"],
            [f"{avg_ram:.2f}%"],
            [f"{high_latency_events}"]
        ]
        
        rows = ['Failed Requests (Total)', 'Average CPU Usage', 'Average Memory Usage', 'High Latency Events']

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=['Valor Obtenido'], loc='center', cellLoc='center')
        
        # Estilos
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        save_path = os.path.join(self.save_dir, 'testing_summary_table.png')
        plt.title('Resumen de Métricas (Fase de Pruebas)', fontweight="bold", pad=20)
        fig.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Tabla de resumen guardada en: {save_path}")