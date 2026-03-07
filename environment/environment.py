import gymnasium as gym
import requests
import time
from gymnasium import spaces
import numpy as np

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
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
            time.sleep(3) # Ajustado para no desincronizar metricas
            self.actual_state = self.get_real_metrics()
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
        
        info = {"activos": cant_active_containers, "step": self.current_step}
        
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

        # Inicializamos las variables de estado la primera vez
        if not hasattr(self, 'sim_traffic_fn') or (self.current_step - self.sim_fn_start_step >= self.sim_fn_duration):
            # Decidimos los tiempos limites y funciones a usar
            self.sim_traffic_fn = np.random.randint(0, 4)
            self.sim_fn_start_step = self.current_step
            self.sim_fn_duration = np.random.randint(50, 200) # Duracion en steps
            
            self.sim_total_users = 450 
            
            # Se generan los valores necesarios para cada funcion
            self.sim_peak_one = self.sim_total_users * np.random.uniform(0.5, 0.9)
            self.sim_min = self.sim_total_users * np.random.uniform(0.05, 0.20)
            self.sim_peak_two = self.sim_total_users * np.random.uniform(0.5, 0.9)
            self.sim_base = self.sim_total_users * np.random.uniform(0.02, 0.15)
            self.sim_scale_rate = (self.sim_total_users - self.sim_base) / (self.sim_fn_duration / 2)

        relative_step = self.current_step - self.sim_fn_start_step
        mid_step = self.sim_fn_duration / 2
        workload = 10.0 # Valor por defecto

        # 0: Double Wave
        if self.sim_traffic_fn == 0:
            import math
            workload = (
                (self.sim_peak_one - self.sim_min) * math.e ** -(((relative_step / (self.sim_fn_duration / 10 * 2 / 3)) - 5) ** 2)
                + (self.sim_peak_two - self.sim_min) * math.e ** -(((relative_step / (self.sim_fn_duration / 10 * 2 / 3)) - 10) ** 2)
                + self.sim_min
            )
            
        # 1: Lineal
        elif self.sim_traffic_fn == 1:
            if relative_step <= mid_step:
                workload = self.sim_base + (self.sim_scale_rate * relative_step)
            else:
                peak = self.sim_base + (self.sim_scale_rate * mid_step)
                time_down = relative_step - mid_step
                workload = peak - (self.sim_scale_rate * time_down)

        # 2: Exponencial
        elif self.sim_traffic_fn == 2:
            import math
            peak_time = self.sim_fn_duration * 0.7
            base_users = max(10, self.sim_base)
            if relative_step <= peak_time:
                k = math.log(self.sim_total_users / base_users) / max(1, peak_time)
                workload = base_users * math.exp(k * relative_step)
            else:
                time_down = relative_step - peak_time
                drop_rate = (self.sim_total_users - base_users) / max(1, self.sim_fn_duration - peak_time)
                workload = self.sim_total_users - (drop_rate * time_down)

        # 3: Step (Escalones)
        elif self.sim_traffic_fn == 3:
            step_size = max(5, self.sim_fn_duration // 10)
            users_per_step = (self.sim_total_users - self.sim_base) / 5
            if relative_step <= mid_step:
                current_step_idx = relative_step // step_size
                workload = self.sim_base + (users_per_step * current_step_idx)
            else:
                peak_steps = mid_step // step_size
                peak_users = self.sim_base + (users_per_step * peak_steps)
                steps_down = (relative_step - mid_step) // step_size
                workload = peak_users - (users_per_step * steps_down)

        return max(10, min(workload, self.sim_total_users))

    def get_simulated_metrics(self, action):
        """
        TODO: modelo matemático que simula la CPU y Latencia 
        """

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
                # Capacidad del nodo 50 'unidades' de trabajo
                node_load = total_workload * norm_weights[i]
                cpu_usage = min(1.0, node_load / 50.0) 
                
               
                latency_ms = 10 + (cpu_usage ** 4) * 500 
                
                errors = max(0.0, (cpu_usage - 0.9) * 10) if cpu_usage > 0.9 else 0.0
                
                new_state[idx:idx+6] = [
                    cpu_usage,       # cpu_usg
                    0.3,             # ram_usg_pct
                    1.0,             # ram_total_normalize
                    min(1.0, latency_ms / 2000.0), # latency normalizada a 2000ms
                    errors,          # error_rate
                    1.0              # status (ACTIVO)
                ]
            else:
                new_state[idx:idx+6] = [0.0] * 6 # Nodo apagado

        return new_state

    def reward_function(self, state, action, cant_active_containers):

        total_reward = 0.0

        # Pesos re-ajustados 
        W_LATENCY = 2.0      
        W_ERRORS = 50.0      
        W_COST = 1.0         
        W_SATURATION = 1.0   
        
        if cant_active_containers == 0:
            return -200.0 
            
        avg_latency = 0.0
        total_errors = 0.0
        
        for i in range(self.n_max):
            idx_base = i * 6 
            status = state[idx_base + 5] #
            
            if status == 1.0:
                cpu_pct = state[idx_base]
                ram_pct = state[idx_base + 1]
                latency = state[idx_base + 3]
                errores = state[idx_base + 4]

                avg_latency += latency
                total_errors += errores
                
                if cpu_pct > 0.80:
                    total_reward -= W_SATURATION * (cpu_pct - 0.80)
                    
        avg_latency /= cant_active_containers
        
        # Calculo penalizaciones
        latency_penalty = W_LATENCY * (avg_latency ** 2) # Penalizacion exponencial
        error_penalty = W_ERRORS * total_errors
        cost_penalty = W_COST * (cant_active_containers / self.n_max)
        
        total_reward -= (latency_penalty + error_penalty + cost_penalty)

        # Evitar bucle de prendido y apagado
        scale_decision = action[-1]
        if scale_decision <= 0.3 or scale_decision >= 0.7:
            # Pequeña penalidad por ejecutar la acción de escalar
            total_reward -= 0.05 
        
        return total_reward