import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LoadBalancerEnv(gym.Env):
    """
    Entorno PPO para Balanceo de Carga y Auto-scaling (LBDRL)
    """
    
    def __init__(self, n_max=10, max_steps=100, max_memory=1024):
        super(LoadBalancerEnv, self).__init__()
        self.n_max = n_max
        self.max_memory = max_memory 
        self.max_steps = max_steps
        self.current_step = 0

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
        
        self.active_containers = np.zeros(self.n_max, dtype=bool)
        self.actual_state = np.zeros(self.n_max * 6, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reinicia el entorno, apagando el cluster y dejando 1 solo nodo activo"""
        super().reset(seed=seed)
        
        self.active_containers = np.zeros(self.n_max, dtype=bool)
        self.active_containers[0] = True # Nodo principal siempre vivo al inicio
        self.current_step = 0

        # TODO: Llamada al backend para hacer un reset real
        # ej: request.post("http://localhost:5000/api/cluster/reset")
        
        self.actual_state = self.get_metrics()
        info = {"mensaje": "Cluster reiniciado a 1 instancia"}
        
        return self.actual_state, info

    def step(self, action):
        """Avanza un instante de tiempo en el MDP"""
        self.current_step += 1

        raw_weights = action[:self.n_max]
        scale_desition = action[-1]
        
        
        if scale_desition > 0.7: # <- se puede ajustar
            self.scale_up()
        elif scale_desition < 0.3: # <- se puede ajustar
            self.scale_down()
            
        # Balanceo de Carga
        self.re_balance_traffick(raw_weights)
        

        # Esperar / Procesar 
        # TODO: Simulación de tiempo de procesamiento o espera real
        

        self.actual_state = self.get_metrics()
        
        reward = self.reward_function(self.actual_state, action)
        
        # Condiciones de término (todos los contenedores cayeron)
        terminated = not np.any(self.active_containers)
        
        if self.current_step >= self.max_steps:
            truncated = True 
        else:
            truncated = False
        
        info = {
                "activos": np.sum(self.active_containers),
                "step": self.current_step
                }
        
        return self.actual_state, reward, terminated, truncated, info

    def scale_up(self):
        """Encuentra el primer slot libre y levanta un contenedor"""
        off_containers = np.where(~self.active_containers)[0]
        if len(off_containers) > 0:
            idx_nuevo = off_containers[0]
            self.active_containers[idx_nuevo] = True
            # TODO: Llamada a Docker SDK para encender contenedor idx_nuevo
            # print(f"[Scaler] Levantando nodo {idx_nuevo}")

    def scale_down(self):
        """Apaga el último contenedor activo (protegiendo el nodo 0)"""
        on_containers = np.where(self.active_containers)[0]
        if len(on_containers) > 1: # Prevenimos apagar todo el cluster
            idx_apagar = on_containers[-1]
            self.active_containers[idx_apagar] = False
            # TODO: Llamada a Docker SDK para detener contenedor idx_apagar
            # print(f"[Scaler] Apagando nodo {idx_apagar}")

    def re_balance_traffick(self, raw_weights):
        """Enmascara nodos muertos, normaliza pesos y reconfigura el proxy"""

        # Aplicamos enmascaramiento -> nodos apagados = 0
        filtered_weights = raw_weights * self.active_containers
        
        sum_weights = np.sum(filtered_weights)
        if sum_weights > 0:
            final_weights = filtered_weights / sum_weights
        else:
            # Fallback a Round Robin puro entre los nodos activos.
            final_weights = np.zeros(self.n_max)
            activos = np.where(self.active_containers)[0]
            final_weights[activos] = 1.0 / len(activos)
            
        final_weights = np.round(final_weights, 4)
        
        # TODO: Actualizar pesos en Nginx via API o reescribiendo haproxy.cfg
        # print(f"[Proxy] Nuevos pesos enrutamiento: {pesos_finales}")

    def get_metrics(self):
        """Consulta el estado de la infraestructura y arma el vector de observaciones"""

        new_state = np.zeros(self.n_max * 6, dtype=np.float32)
        
        for i in range(self.n_max):
            if self.active_containers[i]:
                # TODO: Reemplazar con lectura real de Docker SDK

                # Simulación:
                cpu_pct = np.random.uniform(0.1, 0.9)
                ram_pct = np.random.uniform(0.2, 0.8)
                
                # Supongamos que este contenedor tiene 512MB de límite real
                ram_total_mb = 512.0 
                
                # Normalizamos dividiendo por el máximo permitido en el clúster
                ram_total_norm = ram_total_mb / self.max_memory 
                
                response_time = np.random.uniform(0.01, 0.15)
                errores = 0.0
                status = 1.0
                
                # Mapear las 6 métricas al vector plano
                idx_base = i * 6
                new_state[idx_base : idx_base+6] = [
                    cpu_pct, 
                    ram_pct, 
                    ram_total_norm, 
                    response_time, 
                    errores, 
                    status
                ]
            else:
                # Los nodos apagados mantienen sus 6 valores en 0.0
                pass 
                
        return new_state

    def reward_function(self, state, action):
        """
        Calcula la recompensa evaluando latencia, errores, costo de 
        contenedores y salud individual de cada nodo
        """

        total_reward = 0.0
        
        # Pesos 
        W_LATENCY = 1.0
        W_ERRORS = 10.0      # Errores inaceptables -> alto peso
        W_COST = 0.2         # Queremos ahorrar, pero no somos ratas 
        W_SATURATION = 0.5    # Penalizacion por estresar demasiado un solo nodo
        
        cant_active_containers = np.sum(self.active_containers)
        
        # El cluster esta apagado
        if cant_active_containers == 0:
            return -100.0 
            
        avg_latency = 0.0
        total_errors = 0.0
        
        # Recorremos el vector de estado
        for i in range(self.n_max):
            if self.active_containers[i]:
                idx_base = i * 6 # Cada nodo tiene 6 metricas consecutivas
                cpu_pct = state[idx_base]
                ram_pct = state[idx_base + 1]
                latency = state[idx_base + 3]
                errores = state[idx_base + 4]

                avg_latency += latency
                total_errors += errores
                
                # Riesgo de Saturación
                # Si pasa del 85% de uso -> penalización exponencial
                if cpu_pct > 0.85:
                    total_reward -= W_SATURATION * ((cpu_pct - 0.85) * 10)
                if ram_pct > 0.85:
                    total_reward -= W_SATURATION * ((ram_pct - 0.85) * 10)
                    
        avg_latency /= cant_active_containers
        
        # Pesos Latencia y Errores
        penalizacion_latencia = W_LATENCY * avg_latency
        penalizacion_errores = W_ERRORS * total_errors
        
        # Costo de Infraestructura
        # 1 nodo -> poco castigo | 10 nodos -> castigo máximo
        costo_infra = W_COST * (cant_active_containers / self.n_max)
        

        # Recompensa Total = Recompensa Base (0) - Penalizaciones
        total_reward -= (penalizacion_latencia + penalizacion_errores + costo_infra)
        

        # Evitar bucle de prendido y apagado
        scale_decision = action[-1]
        if scale_decision < 0.3 or scale_decision > 0.7:
             # Pequeña penalidad por ejecutar la acción de escalar
            total_reward -= 0.05 
            
        return total_reward