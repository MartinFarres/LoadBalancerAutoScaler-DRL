from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import time

def run_industry_baseline(simulated=True, steps=5000):
    print("Iniciando prueba del Baseline de la Industria (Round Robin + Thresholds)...")
    
    env = LoadBalancerEnv(simulated=simulated, max_steps=steps)
    obs, info = env.reset()
    
    # Auto Scaler Tradicional
    CPU_THRESHOLD_UP = 0.75   # Escalar si CPU > 75%
    CPU_THRESHOLD_DOWN = 0.25 # Desescalar si CPU < 25%
    
    hist_cpu_total = [] 
    hist_ram_total = [] 
    hist_latency = []   
    hist_errors = []
    
    # Inicializamos el visualizador 
    viz = Visualizer(save_dir="./resultados_graficos/baseline")

    for i in range(steps):
        activos = info.get('activos', 1)
        
        cpu_total = 0.0
        ram_total = 0.0
        avg_latency = 0.0
        total_errors = 0.0
        
        for j in range(activos):
            cpu_total += obs[j * 6]        # CPU
            ram_total += obs[j * 6 + 1]    # RAM
            avg_latency += obs[j * 6 + 3]  # Latency
            total_errors += obs[j * 6 + 4] # Error Rate
            
        avg_cpu = cpu_total / activos if activos > 0 else 0.0
        
        # Guardar telemetría
        hist_cpu_total.append(avg_cpu)
        hist_ram_total.append(ram_total / activos if activos > 0 else 0)
        hist_latency.append(avg_latency / activos if activos > 0 else 0)
        hist_errors.append(total_errors)
        
        #  Baseline Auto Scaling 
        scale_decision = 0.5
        
        if avg_cpu >= CPU_THRESHOLD_UP:
            scale_decision = 1.0 # Scale Up
        elif avg_cpu <= CPU_THRESHOLD_DOWN:
            scale_decision = 0.0 # Scale Down
            
        # Baseline Load Balancing (Round Robin / Pesos Equitativos)
        # Asignamos peso 1.0 a todos los nodos. HAProxy balancea equitativamente.
        weights = [1.0] * env.n_max 
        
        # Formatear la acción para el entorno
        action = np.array(weights + [scale_decision], dtype=np.float32)
        
        # Ejecutar acción
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

    print("Generando tabla resumen del Baseline...")
    viz.generate_testing_summary_table(
        cpu_history=hist_cpu_total,
        ram_history=hist_ram_total,
        latency_history=hist_latency,
        errors_history=hist_errors,
        high_latency_threshold=0.8 
    )

if __name__ == "__main__":
    run_industry_baseline(simulated=True)