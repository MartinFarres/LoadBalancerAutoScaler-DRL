import gymnasium as gym
import requests
import time
from gymnasium import spaces
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.traffic_generator import TrafficGenerator

class LoadBalancerEnv(gym.Env):
    """
    Entorno PPO para Balanceo de Carga y Auto-scaling (LBDRL)
    Soporta modo Real (FastAPI/Docker) y modo Simulado (Matemático)
    """
    
    def __init__(self, n_max=10, max_steps=100, max_memory=1024, api_url="http://127.0.0.1:8000", simulated=False):
        super(LoadBalancerEnv, self).__init__()
        self.n_max = n_max
        self.max_memory = max_memory 
        self.max_steps = max_steps
        self.current_step = 0
        self.api_url = api_url
        self.simulated = simulated

        # OBSERVATION SPACE
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max * 6,), 
            dtype=np.float32
        )
        
        # ACTION SPACE
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max + 1,), 
            dtype=np.float32
        )
        
        self.actual_state = np.zeros(self.n_max * 6, dtype=np.float32)

        # Variables exclusivas para el modo simulado
        if self.simulated:
            self.current_step = 0
            self.sim_active_containers = np.zeros(self.n_max, dtype=bool)
            self.sim_active_containers[0] = True

        # Memoria para la Espera Dinamica
        self.last_weights = np.zeros(self.n_max, dtype=np.float32)

        # Variables exclusivas para el modo simulado
        if self.simulated:
            self.current_step = 0
            self.sim_active_containers = np.zeros(self.n_max, dtype=bool)
            self.sim_active_containers = True
            
            self.traffic_gen = TrafficGenerator()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_weights = np.zeros(self.n_max, dtype=np.float32) # Reiniciamos la memoria
        
        if not self.simulated:
            requests.get(f"{self.api_url}/reset")
            time.sleep(0.5) # Ajustado al límite para HAProxy
            self.actual_state = self.get_real_metrics()
        else:
            # Reseteo del estado simulado
            self.sim_active_containers = np.zeros(self.n_max, dtype=bool)
            self.sim_active_containers[0] = True
            self.actual_state = self.get_simulated_metrics(action=None)

        info = {"mensaje": f"Cluster reiniciado a 1 instancia (Simulado: {self.simulated})"}
        return self.actual_state, info

    def step(self, action):
        self.current_step += 1

        raw_weights = action[:self.n_max]
        scale_desision = action[-1]
        
        if not self.simulated:
            payload = {
                "weights": raw_weights.tolist(),
                "decision": float(scale_desision) 
            }
            requests.post(f"{self.api_url}/action", json=payload)

            # ESPERA DINAMICA
            # Escalo el cluster?
            is_scaling = scale_desision >= 0.7 or scale_desision <= 0.3
            
            # Cambio de pesos drastico?
            weight_diff = np.max(np.abs(raw_weights - self.last_weights))
            is_shifting_traffic = weight_diff > 0.15
            
            if is_scaling:
                # Cambio fisico: HAProxy se reinicia y Docker levanta/baja contenedores.
                time.sleep(3.0) 
            elif is_shifting_traffic:
                # Cambio logico: HAProxy redirige tráfico, necesitamos que la latencia y CPU se actualicen.
                time.sleep(1.5)
            else:
                # Estado de reposo: El agente no hizo nada. Solo leemos el siguiente tick de CPU.
                time.sleep(0.4) 
                
            self.actual_state = self.get_real_metrics()
            
            # Guardamos los pesos actuales para compararlos en el proximo step
            self.last_weights = np.copy(raw_weights)
        else:
            # Actualizar estado simulado
            #self.actual_state = self.get_simulated_metrics(action)
            if scale_desision >= 0.7:
                for i in range(self.n_max):
                    if not self.sim_active_containers[i]:
                        self.sim_active_containers[i] = True
                        break
            elif scale_desision <= 0.3:
                for i in range(self.n_max - 1, 0, -1): # Recorre desde el final hasta el penultimo asegurando al menos 1 activo
                    if self.sim_active_containers[i]:
                        self.sim_active_containers[i] = False
                        break
            
            self.actual_state = self.get_simulated_metrics(action)
        
        # Calcular contenedores vivos leyendo el status del estado actual
        cant_active_containers = sum(1 for i in range(self.n_max) if self.actual_state[(i * 6) + 5] == 1.0)
        
        reward = self.reward_function(self.actual_state, action, cant_active_containers)
        
        terminated = (cant_active_containers == 0)
        truncated = (self.current_step >= self.max_steps)

        # Sacarmos promedio de info de nodos
        cpu_t = ram_t = lat_t = err_t = 0.0
        if cant_active_containers > 0:
            for i in range(self.n_max):
                idx = i * 6
                if self.actual_state[idx + 5] == 1.0: # Si está ACTIVO
                    cpu_t += self.actual_state[idx]
                    ram_t += self.actual_state[idx+1]
                    lat_t += self.actual_state[idx+3]
                    err_t += self.actual_state[idx+4]
                    
            cpu_t /= cant_active_containers
            ram_t /= cant_active_containers
            lat_t /= cant_active_containers
            err_t /= cant_active_containers
        

        info = {
            "activos": cant_active_containers, 
            "step": self.current_step,
            "cpu_avg": float(cpu_t),
            "ram_avg": float(ram_t),
            "latency_avg": float(lat_t),
            "error_avg": float(err_t)
        }
        
        return self.actual_state, reward, terminated, truncated, info

    def get_real_metrics(self):
        new_state = []
        try:
            response = requests.get(f"{self.api_url}/metrics").json()
            
            MAX_LATENCY_MS = 2000.0 # 2 segundos máximo
            
            for i in range(self.n_max):
                
                cpu_raw = response[i]["cpu_usg"]
                cpu_norm = min(1.0, max(0.0, cpu_raw))
                
                ram_raw = response[i]["ram_usg_pct"]
                ram_norm = min(1.0, max(0.0, ram_raw))
                
                ram_tot_raw = response[i]["ram_total_normalize"]
                ram_tot_norm = min(1.0, max(0.0, ram_tot_raw))
                
                # Convertimos milisegundos a una escala 0.0 - 1.0
                latency_raw = response[i]["latency"]
                latency_norm = min(1.0, latency_raw / MAX_LATENCY_MS)
                
                error_raw = response[i]["error_rate"]
                error_norm = min(1.0, max(0.0, error_raw))
                
                status = float(response[i]["status"])
                
                new_state.extend([cpu_norm, ram_norm, ram_tot_norm, latency_norm, error_norm, status])
                
        except Exception as e:
            print(f"Error leyendo API: {e}")
            new_state = [0.0] * (self.n_max * 6)
            new_state[5] = 1.0 
            
        return np.array(new_state, dtype=np.float32)

    def get_dynamic_simulated_workload(self):
        # Obtenemos la carga pasándole el step actual de la simulación
        workload = self.traffic_gen.get_workload(self.current_step)
        return workload

    def get_simulated_metrics(self, action):
        total_workload = self.get_dynamic_simulated_workload()
        
        new_state = np.zeros(self.n_max * 6, dtype=np.float32)

        # Distribución de carga según pesos de la acción
        raw_weights = action[:self.n_max] if action is not None else np.zeros(self.n_max)

        # Ignoro pesos de nodos apagados
        active_mask = self.sim_active_containers.astype(float)
        effective_weights = raw_weights * active_mask
        
        sum_w = np.sum(effective_weights)
        norm_weights = effective_weights / sum_w if sum_w > 0 else effective_weights

        if sum_w == 0 and self.sim_active_containers[0]: norm_weights[0] = 1.0

        for i in range(self.n_max):
            idx = i * 6
            if self.sim_active_containers[i]:
                # M/M/1 -----------------------------------------------------------------------------------------
                
                # Params Base
                max_capacity = 50.0  # mu: Tasa de servicio: peticiones maximas tericas por paso
                base_latency_ms = 15.0 # S: Tiempo de servicio base sin hacer cola
                
                # Tasa de llegada (lambda)
                node_load = total_workload * norm_weights[i]
                
                # Utilización del Sistema (rho = lambda / mu)
                rho = node_load / max_capacity 
                
                # CALCULO DE CPU ------------------------------------------------
                # Se añade ruido gaussiano 
                cpu_noise = np.random.normal(0, 0.02)
                cpu_usage = max(0.0, min(1.0, rho + cpu_noise))
                
                # CALCULO DE LATENCIA ----------------------------------------------
                # Basado en el teorema M/M/1: Tiempo de Respuesta E[T] = S / (1 - rho)
                if rho < 0.95:
                    # la cola crece exponencialmente al acercarse al 100%
                    latency_ms = base_latency_ms / (1.0 - rho)
                else:
                    # Si el sistema está saturado (rho >= 0.95), la cola "explota" y se degrada fuertemente.
                    latency_ms = base_latency_ms * 20.0 + (rho - 0.95) * 5000.0
                
                # Jitter de red (ruido en la latencia)
                latency_ms += abs(np.random.normal(0, latency_ms * 0.05))
                
                # CALCULO DE ERRORES ---------------------------------------------
                # Simulamos un desbordamiento de bufer (Queue overflow). 
                # Usamos una función sigmoide para que la tasa de errores pase de 0% a 100% de forma suave 
                # pero rápida cuando la capacidad supera el 100% (rho > 1.0)
                if rho < 0.9:
                    errors = 0.0
                else:
                    # 1 / (1 + e^(-k*(x-x0)))
                    errors = 1.0 / (1.0 + math.exp(-15.0 * (rho - 1.05)))


                # CALCULO DE RAM (Ley de Little) --------------------------------
                # L: El num de peticiones concurrentes dentro del contenedor (procesandose + en cola)
                if rho < 0.95:
                    L_concurrent = rho / (1.0 - rho)
                else:
                    # Si el sistema colapsa, la acumulación de peticiones en la RAM se dispara
                    L_concurrent = (0.95 / 0.05) + (rho - 0.95) * 200.0
                
                # Definimos la huella de memoria
                base_ram_footprint = 0.15 # 15% de RAM ocupada solo por tener el contenedor encendido (Python/Flask)
                ram_per_request = 0.015   # Cada petición viva en el sistema consume un 1.5% adicional de RAM
                
                ram_usage = base_ram_footprint + (L_concurrent * ram_per_request)
                
                # Ruido simulando el Garbage Collector de Python (liberando y ocupando memoria)
                ram_noise = np.random.normal(0, 0.02)
                ram_usage = min(1.0, max(0.0, ram_usage + ram_noise))
                
                new_state[idx:idx+6] = [
                    min(1.0, max(0.0, cpu_usage)),             # cpu_usg
                    ram_usage,                                 # ram_usg_pct 
                    1.0,                                       # ram_total_normalize
                    min(1.0, latency_ms / 2000.0),             # latency normalizada a 2000ms
                    min(1.0, max(0.0, errors)),                # error_rate
                    1.0                                        # status (ACTIVO)
                ]
            else:
                new_state[idx:idx+6] = [0.0] * 6 # Nodo apagado

        return new_state

    def reward_function(self, state, action, cant_active_containers):

        total_reward = 0.0

        W_LATENCY = 2.0      
        W_ERRORS = 50.0      
        W_COST = 1.0         
        W_SATURATION = 1.0   
        W_RAM_SATURATION = 1.0 
        
        if cant_active_containers == 0:
            return -200.0 
            
        # Variables para acumular
        avg_latency = 0.0
        avg_errors = 0.0
        avg_cpu_sat = 0.0
        avg_ram_sat = 0.0
        
        for i in range(self.n_max):
            idx_base = i * 6 
            status = state[idx_base + 5] 
            
            if status == 1.0:
                cpu_pct = state[idx_base]
                ram_pct = state[idx_base + 1]
                latency = state[idx_base + 3]
                errores = state[idx_base + 4]

                avg_latency += latency
                avg_errors += errores
                
                if cpu_pct > 0.80:
                    avg_cpu_sat += (cpu_pct - 0.80)
                if ram_pct > 0.85:
                    avg_ram_sat += (ram_pct - 0.85)
                    
        # PROMEDIAMOS las métricas de los nodos encendidos
        avg_latency /= cant_active_containers
        avg_errors /= cant_active_containers
        avg_cpu_sat /= cant_active_containers
        avg_ram_sat /= cant_active_containers
        
        # Calculo de penalizaciones (ahora usando los promedios reales)
        latency_penalty = W_LATENCY * (avg_latency ** 2) 
        error_penalty = W_ERRORS * avg_errors 
        cost_penalty = W_COST * (cant_active_containers / self.n_max)
        saturation_penalty = (W_SATURATION * avg_cpu_sat) + (W_RAM_SATURATION * avg_ram_sat)
        
        total_reward -= (latency_penalty + error_penalty + cost_penalty + saturation_penalty)

        # Evitar bucle de prendido y apagado
        scale_decision = action[-1]
        if scale_decision <= 0.3 or scale_decision >= 0.7:
            total_reward -= 0.05 
        
        return total_reward