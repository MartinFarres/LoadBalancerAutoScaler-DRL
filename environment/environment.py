import gymnasium as gym
import requests
import time
from gymnasium import spaces
import numpy as np

class LoadBalancerEnv(gym.Env):
    """
    Entorno PPO para Balanceo de Carga y Auto-scaling (LBDRL)
    """
    
    def __init__(self, n_max=10, max_steps=100, max_memory=1024, api_url="http://127.0.0.1:8000"):
        super(LoadBalancerEnv, self).__init__()
        self.n_max = n_max
        self.max_memory = max_memory 
        self.max_steps = max_steps
        self.current_step = 0
        self.api_url = api_url

        # OBSERVATION SPACE
        # Por cada contenedor: [CPU, RAM, MEM, Tiempo_Respuesta, Tasa_Errores, Is_Active] 
        # Todos los valores normalizados entre 0.0 y 1.0
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max * 6,), 
            dtype=np.float32
        )
        
        # ACTION SPACE
        # N_max valores para pesos de ruteo + 1 valor para decisión de auto-scaling. 
        # [w1, w2, ..., wN, scale_desition]
        # Rango 0.0 a 1.0.
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max + 1,), 
            dtype=np.float32
        )
        
        self.actual_state = np.zeros(self.n_max * 6, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reinicia el entorno, apagando el cluster y dejando 1 solo nodo activo"""
        super().reset(seed=seed)
        self.current_step = 0
        
        requests.get(f"{self.api_url}/reset")
        
        # Damos tiempo a la infraestructura para estabilizarse tras el reset
        time.sleep(1)
        
        self.actual_state = self.get_metrics()
        info = {"mensaje": "Cluster reiniciado a 1 instancia"}
        
        return self.actual_state, info

    def step(self, action):
        """Avanza un instante de tiempo en el MDP"""
        self.current_step += 1

        raw_weights = action[:self.n_max]
        scale_desition = action[-1]
        
        payload = {
            "weights": raw_weights.tolist(),
            "decision": float(scale_desition) 
        }
        
        # Ejecutar la accion en el cluster real
        requests.post(f"{self.api_url}/action", json=payload)
        
        # Esperar a que el proxy enrute trafico y genere metricas
        time.sleep(1) 
       
        # Observar el nuevo estado
        self.actual_state = self.get_metrics()
        
        # Calcular contenedores vivos leyendo el status
        cant_active_containers = sum(1 for i in range(self.n_max) if self.actual_state[(i * 6) + 5] == 1.0)
        
        # Calcular recompensa
        reward = self.reward_function(self.actual_state, action, cant_active_containers)
        
        # Condiciones de término
        terminated = (cant_active_containers == 0)
        truncated = (self.current_step >= self.max_steps)
        
        info = {
            "activos": cant_active_containers,
            "step": self.current_step
        }
        
        return self.actual_state, reward, terminated, truncated, info

    def get_metrics(self):
        """Consulta el estado de la infraestructura y arma el vector de observaciones"""
        new_state = []
        response = requests.get(f"{self.api_url}/metrics").json()

        for i in range(self.n_max):
            new_state.append(response[i]["cpu_usg"])
            new_state.append(response[i]["ram_usg_pct"])
            new_state.append(response[i]["ram_total_normalize"])
            new_state.append(response[i]["latency"])
            new_state.append(response[i]["error_rate"])
            new_state.append(response[i]["status"])              
        
        return np.array(new_state, dtype=np.float32)

    def reward_function(self, state, action, cant_active_containers):
        """
        Calcula la recompensa evaluando latencia, errores, costo de 
        contenedores y salud individual de cada nodo
        """
        total_reward = 0.0
        
        # Pesos 
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
            status = state[idx_base + 5] # Leemos la metrica Is_Active
            
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
        
        penalizacion_latencia = W_LATENCY * avg_latency
        penalizacion_errores = W_ERRORS * total_errors
        costo_infra = W_COST * (cant_active_containers / self.n_max)
        
        total_reward -= (penalizacion_latencia + penalizacion_errores + costo_infra)
        
        scale_decision = action[-1]
        if scale_decision < 0.3 or scale_decision > 0.7:
            total_reward -= 0.05 
            
        return total_reward