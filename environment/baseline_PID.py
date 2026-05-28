from environment import LoadBalancerEnv
from visualizer import Visualizer
import numpy as np
import pandas as pd
import os

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp  # Constante Proporcional (reacción inmediata)
        self.ki = ki  # Constante Integral (memoria de errores pasados)
        self.kd = kd  # Constante Derivativa (predicción de inercia)
        self.setpoint = setpoint # Valor objetivo (ej. 60% de CPU)
        
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, current_value):
        # Calculamos cuánto nos desviamos del valor objetivo
        error = current_value - self.setpoint
        
        # Acumulamos el error en el tiempo (Integral)
        self.integral += error
        
        # Calculamos la tasa de cambio del error (Derivativa)
        derivative = error - self.previous_error
        
        # Ecuación estándar del PID
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.previous_error = error
        return output

def run_pid_baseline(simulated=True, steps=5000, n_max=5, file='testing_metrics.csv'):
    print("Iniciando prueba del Baseline PID (Teoría de Control Clásica)...")
    
    env = LoadBalancerEnv(simulated=simulated, max_steps=steps, n_max=n_max, testing=True)
    obs, info = env.reset(42)
    
    # Target: Mantener la CPU promedio al 60% (0.60)
    pid = PIDController(kp=1.5, ki=0.1, kd=0.5, setpoint=0.60)
    
    hist_reward = []
    hist_cpu_total = []
    hist_ram_total = []
    hist_latency = []
    hist_errors = []
    hist_workload = []
    hist_activos = []

     # metricas
    scaling_events = 0
    sla_violations = 0
    last_activos = 1 

    viz = Visualizer(save_dir="./resultados_graficos/baseline_pid")

    for i in range(steps):
        activos = info.get('activos', 1)
        workload = info.get('workload', 0.0) * 4000 # Desnormalizamos -> asumiendo 4000 total_users

        # Conteo de eventos de escalado 
        if i > 0 and activos != last_activos:
            scaling_events += 1
        last_activos = activos

        cpu_total = 0.0
        ram_total = 0.0
        avg_latency = 0.0
        total_errors = 0.0
        
        for j in range(activos):
            cpu_total += obs[j * 6]
            ram_total += obs[j * 6 + 1]
            avg_latency += obs[j * 6 + 3]
            total_errors += obs[j * 6 + 4]
            
        avg_cpu = cpu_total / activos if activos > 0 else 0.0
        
        if activos > 0:
            cpu_total /= activos  # Sacamos el promedio real
            ram_total /= activos  # Sacamos el promedio real 
            avg_latency /= activos
        
        # Conteo de violaciones SLA (latencia > 0.5 )
        if avg_latency > 0.5:
            sla_violations += 1

        # Guardamos
        hist_cpu_total.append(avg_cpu)
        hist_ram_total.append(ram_total)
        hist_latency.append(avg_latency)
        hist_errors.append(total_errors)
        hist_activos.append(activos)
        hist_workload.append(workload)

        # --- LOGICA DE CONTROL PID ---
        # El PID calcula la "señal de control" basandose en la CPU actual
        control_signal = pid.compute(avg_cpu)
        
        scale_decision = 0.5 
        
        if control_signal > 0.4:
            scale_decision = 1.0 # Scale Up
        elif control_signal < -0.4:
            scale_decision = 0.0 # Scale Down
            
        # Round Robin
        weights = [1.0] * env.n_max 
        action = np.array(weights + [scale_decision], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        hist_reward.append(reward)

        if terminated or truncated:
            break
    
    total_steps = len(hist_cpu_total)
    sla_violation_pct = (sla_violations / total_steps) * 100

    # Eficiencia de costo: Usuarios soportados por cada contenedor activo
    cost_efficiencies = [w/max(1, a) for w, a in zip(hist_workload, hist_activos)]
    avg_cost_efficiency = np.mean(cost_efficiencies)
    
  
    mode_tag = "sim" if simulated else "real"
    formatted_file = f"test_pid_{mode_tag}_n{n_max}_i{steps}_{file}"
    save_dir = f"./training_results/testing_results_{n_max}_nodes"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, formatted_file)
    pd.DataFrame({
        'step': range(total_steps),
        'reward': hist_reward,
        'cpu_mean': hist_cpu_total,
        'ram_mean': hist_ram_total,
        'latency_mean': hist_latency,
        'error_mean': hist_errors,
        'workload': [w / 4000 for w in hist_workload],
        'activos': hist_activos
    }).to_csv(save_path, index=False)
    print(f"Métricas del PID guardadas en: {save_path}")

    print("Generando tabla resumen del Baseline PID...")
    viz.generate_testing_summary_table(
        cpu_history=hist_cpu_total,
        ram_history=hist_ram_total,
        latency_history=hist_latency,
        errors_history=hist_errors,
        scaling_events=scaling_events,
        sla_violation_pct=sla_violation_pct,
        avg_cost_efficiency=avg_cost_efficiency,
        mode=mode_tag,
    )

    print("Generando curva de recompensa del Baseline PID...")
    viz.plot_testing_reward_curve(csv_path=save_path, mode=mode_tag)

    print("Generando gráficos de comportamiento de workload del Baseline PID...")
    viz.plot_testing_behavior(csv_path=save_path, n_max=n_max, mode=mode_tag)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--file', type=str, default='testing_metrics.csv')
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--simulated', action='store_true', default=False)
    args = parser.parse_args()

    run_pid_baseline(simulated=args.simulated, steps=args.iterations, n_max=args.nodes, file=args.file)