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
            self.sim_active_containers = np.zeros(self.n_max, dtype=bool)

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
        scale_desition = action[-1]
        
        if not self.simulated:
            payload = {
                "weights": raw_weights.tolist(),
                "decision": float(scale_desition) 
            }
            requests.post(f"{self.api_url}/action", json=payload)
            time.sleep(0.3) # Ajustado para no desincronizar metricas
            self.actual_state = self.get_real_metrics()
        else:
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
            for i in range(self.n_max):
                new_state.append(response[i]["cpu_usg"])
                new_state.append(response[i]["ram_usg_pct"])
                new_state.append(response[i]["ram_total_normalize"])
                new_state.append(response[i]["latency"])
                new_state.append(response[i]["error_rate"])
                new_state.append(response[i]["status"])              
        except Exception:
            # Fallback seguro en caso de que la API parpadee
            new_state = [0.0] * (self.n_max * 6)
            
        return np.array(new_state, dtype=np.float32)

    def get_simulated_metrics(self, action):
        """
        TODO: modelo matemático que simula la CPU y Latencia 
        """
        new_state = np.zeros(self.n_max * 6, dtype=np.float32)
        
        if action is not None:
            pass
            
        return new_state

    def reward_function(self, state, action, cant_active_containers):
        total_reward = 0.0
        
        W_LATENCY = 1.0
        W_ERRORS = 10.0      
        W_COST = 0.2         
        W_SATURATION = 0.5    
        
        if cant_active_containers == 0:
            return -100.0 
            
        avg_latency = 0.0
        total_errors = 0.0
        
        for i in range(self.n_max):
            idx_base = i * 6 
            status = state[idx_base + 5] 
            
            if status == 1.0:
                cpu_pct = state[idx_base]
                ram_pct = state[idx_base + 1]
                latency = state[idx_base + 3]
                errores = state[idx_base + 4]

                avg_latency += latency
                total_errors += errores
                
                if cpu_pct > 0.85:
                    total_reward -= W_SATURATION * ((cpu_pct - 0.85) * 10)
                if ram_pct > 0.85:
                    total_reward -= W_SATURATION * ((ram_pct - 0.85) * 10)
                    
        avg_latency /= cant_active_containers
        
        total_reward -= (W_LATENCY * avg_latency + W_ERRORS * total_errors + W_COST * (cant_active_containers / self.n_max))
        
        scale_decision = action[-1]
        if scale_decision < 0.3 or scale_decision > 0.7:
            total_reward -= 0.05 
            
        return total_reward