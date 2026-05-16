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
        workload = info.get('workload') * 4000 # Desnormalizamos -> asumiendo 4000 total_users

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
        
        if terminated or truncated:
            break
    
    total_steps = len(hist_cpu_total)
    sla_violation_pct = (sla_violations / total_steps) * 100

    # Eficiencia de costo: Usuarios soportados por cada contenedor activo
    cost_efficiencies = [w/max(1, a) for w, a in zip(hist_workload, hist_activos)]
    avg_cost_efficiency = np.mean(cost_efficiencies)
    
  
    formatted_file = f"test_pid_n{n_max}_i{steps}_{file}"
    os.makedirs("./training_results/testing_results", exist_ok=True)
    save_path = os.path.join("./training_results/testing_results", formatted_file)
    pd.DataFrame({
        'cpu_promedio': hist_cpu_total,
        'ram_promedio': hist_ram_total,
        'latencia_promedio': hist_latency,
        'tasa_errores': hist_errors
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
        avg_cost_efficiency=avg_cost_efficiency
    )

if __name__ == "__main__":
    run_pid_baseline(simulated=True)