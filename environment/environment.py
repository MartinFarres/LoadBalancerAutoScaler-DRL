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
from utils.config import MAX_MEMORY, MAX_QUEUE_DEPTH, NODE_CAPACITY, REWARD_WEIGHTS, TOTAL_USERS

# Intervalo de reloj fijo (segundos) entre pasos del agente en modo real. Reemplaza la espera
# dinamica anterior para que la carga de trafico (impulsada por el reloj de Locust) evolucione
# una cantidad consistente por paso. Debe ser >= al tiempo que tarda HAProxy en redistribuir el
# trafico y en producir una ventana de medicion de CPU estable.
STEP_INTERVAL_SECONDS = 1.5

class LoadBalancerEnv(gym.Env):
    """
    Entorno PPO para Balanceo de Carga y Auto-scaling (LBDRL)
    Soporta modo Real (FastAPI/Docker) y modo Simulado (Matemático)
    """
    
    def __init__(self, n_max=10, max_steps=100, max_memory=MAX_MEMORY, api_url="http://127.0.0.1:8000", simulated=False, testing=False, reward_weights=None, total_users=None):
        super(LoadBalancerEnv, self).__init__()
        self.n_max = n_max
        self.max_memory = max_memory
        self.max_steps = max_steps
        self.current_step = 0
        self.api_url = api_url
        self.simulated = simulated
        self.testing = testing

        # Volumen de trafico simulado: default = utils.config.TOTAL_USERS, sobreescribible para que
        # la busqueda robusta multi-escala (utils/sensitivity_analysis.py, Etapa 2) pueda asignarle a
        # cada tamano de flota el total_users que le corresponde (ver utils.config.USERS_PER_NODE).
        self.total_users = total_users if total_users is not None else TOTAL_USERS

        # Pesos de la funcion de recompensa: default = utils.config.REWARD_WEIGHTS, sobreescribible
        # por sample al correr el analisis de sensibilidad Saltelli/Sobol (ver utils/sensitivity_analysis.py).
        self.reward_weights = dict(REWARD_WEIGHTS)
        if reward_weights:
            unknown = set(reward_weights) - set(REWARD_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown reward_weights keys: {unknown}")
            self.reward_weights.update(reward_weights)

        # OBSERVATION SPACE
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max * 6 + 1,), 
            dtype=np.float32
        )
        
        # ACTION SPACE
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max + 1,), 
            dtype=np.float32
        )
        
        self.actual_state = np.zeros(self.n_max * 6 + 1, dtype=np.float32)

        # Memoria para la Espera Dinamica
        self.last_weights = np.zeros(self.n_max, dtype=np.float32)
        self.last_scale_direction: int = 0  # -1 = down, 0 = hold, 1 = up

        # Variables exclusivas para el modo simulado
        if self.simulated:
            self.sim_active_containers = np.zeros(self.n_max, dtype=bool)
            self.sim_active_containers[0] = True
            self.traffic_gen = TrafficGenerator(testing=self.testing, total_users=self.total_users)
            # Reloj de trafico CONTINUO (no se reinicia por episodio). Antes el ciclo de trafico
            # se reiniciaba en cada reset() y la carga se leia con current_step (que vuelve a 0),
            # asi que cada episodio solo veia el ~20% inicial (zona baja) de la funcion y nunca
            # alcanzaba load_max. Con un reloj continuo cada episodio arranca en una fase distinta
            # del ciclo (incluido el pico), igual que el Locust real que corre sin parar.
            self.traffic_clock = 0
            self._traffic_seeded = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_weights = np.zeros(self.n_max, dtype=np.float32) # Reiniciamos la memoria
        self.last_scale_direction = 0
        
        if not self.simulated:
            try:
                requests.get(f"{self.api_url}/reset", timeout=10)
            except Exception as e:
                print(f"Reset request failed: {e}")
            time.sleep(0.5) # Ajustado al límite para HAProxy
            self.actual_state = self.get_real_metrics()
        else:
            # Reseteo del estado simulado
            self.sim_active_containers = np.zeros(self.n_max, dtype=bool)
            self.sim_active_containers[0] = True
            # Solo sembramos el RNG / iniciamos el ciclo la PRIMERA vez (reproducibilidad).
            # En episodios posteriores NO reiniciamos el ciclo: el reloj continuo deja que el
            # trafico avance sin parar, asi distintos episodios ven distintas fases (y picos).
            if not self._traffic_seeded:
                self.traffic_gen.reset(seed=seed)
                self.traffic_clock = 0
                self._traffic_seeded = True
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
            try:
                requests.post(f"{self.api_url}/action", json=payload, timeout=10)
            except Exception as e:
                print(f"Action request failed: {e}")

            # CADENCIA FIJA POR PASO
            # Antes la espera era dinamica (3.0 / 1.5 / 0.4s segun la accion), lo que hacia que
            # cada paso consumiera una cantidad distinta de tiempo de reloj. Como el generador de
            # trafico real avanza con el reloj de Locust (wall-clock), eso provocaba que la carga
            # saltara cantidades variables por paso: el agente entrenado en sim (delta constante
            # por paso) no podia seguirla. Fijamos un intervalo unico para que el delta de carga
            # por paso sea consistente y mas cercano al de la simulacion.
            time.sleep(STEP_INTERVAL_SECONDS)

            self.actual_state = self.get_real_metrics()
            
            # Guardamos los pesos actuales para compararlos en el proximo step
            self.last_weights = np.copy(raw_weights)
        else:
            # Avanzamos el reloj de trafico continuo (no se reinicia entre episodios).
            self.traffic_clock += 1
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

        curr_dir = 1 if scale_desision >= 0.7 else (-1 if scale_desision <= 0.3 else 0)
        if curr_dir != 0:
            self.last_scale_direction = curr_dir

        terminated = (cant_active_containers == 0)
        truncated = (self.current_step >= self.max_steps)

        # Sacarmos promedio de info de nodos
        cpu_t = ram_t = queue_t = lat_t = err_t = 0.0
        if cant_active_containers > 0:
            for i in range(self.n_max):
                idx = i * 6
                if self.actual_state[idx + 5] == 1.0: # Si está ACTIVO
                    cpu_t += self.actual_state[idx]
                    ram_t += self.actual_state[idx+1]
                    queue_t += self.actual_state[idx+2]
                    lat_t += self.actual_state[idx+3]
                    err_t += self.actual_state[idx+4]

            cpu_t /= cant_active_containers
            ram_t /= cant_active_containers
            queue_t /= cant_active_containers
            lat_t /= cant_active_containers
            err_t /= cant_active_containers


        info = {
            "activos": cant_active_containers,
            "step": self.current_step,
            "cpu_avg": float(cpu_t),
            "ram_avg": float(ram_t),
            "queue_avg": float(queue_t),
            "latency_avg": float(lat_t),
            "error_avg": float(err_t),
            "workload": float(self.actual_state[-1]) #add workload info to log
        }
        
        return self.actual_state, reward, terminated, truncated, info

    def get_real_metrics(self):
        new_state = []
        workload_norm = 0.0
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=10).json()

            workload_norm = float(response.get("workload_norm", 0.0))
            nodes = response.get("nodes", response)

            MAX_LATENCY_MS = 1000.0 # 1 segundo máximo (debe coincidir con la normalización de latencia simulada)

            # La cola es fleet-wide (qcur global del backend) replicada en cada nodo activo.
            # Normalizamos por (n_max * MAX_QUEUE_DEPTH): denominador fijo de capacidad máxima de la flota.
            # No varía con n_active, así que queue_norm solo cae cuando el qcur real disminuye,
            # no por el mero hecho de encender un nodo (señal honesta para el agente).
            queue_denom = self.n_max * MAX_QUEUE_DEPTH

            for i in range(self.n_max):
                
                cpu_raw = nodes[i]["cpu_usg"]
                
                cpu_norm = min(1.0, max(0.0, cpu_raw))
                
                ram_raw = nodes[i]["ram_usg_pct"]
                ram_norm = min(1.0, max(0.0, ram_raw))

                # Cola fleet-wide (qcur del backend) normalizada por (n_activos * MAX_QUEUE_DEPTH)
                queue_raw = nodes[i]["queue_depth"]
                queue_norm = min(1.0, max(0.0, queue_raw / queue_denom))

                # Convertimos milisegundos a una escala 0.0 - 1.0
                latency_raw = nodes[i]["latency"]
                latency_norm = min(1.0, latency_raw / MAX_LATENCY_MS)

                error_raw = nodes[i]["error_rate"]
                error_norm = min(1.0, max(0.0, error_raw))

                status = float(nodes[i]["status"])

                new_state.extend([cpu_norm, ram_norm, queue_norm, latency_norm, error_norm, status])
                
        except Exception as e:
            print(f"Error leyendo API: {e}")
            new_state = [0.0] * (self.n_max * 6)
            new_state[5] = 1.0 
            
        # agregamos workload 
        new_state.append(workload_norm)
        
        return np.array(new_state, dtype=np.float32)

    def get_dynamic_simulated_workload(self):
        # Usamos el reloj de trafico CONTINUO (no current_step, que se reinicia por episodio)
        # para que el ciclo de trafico avance sin parar y los episodios cubran todas sus fases.
        workload = self.traffic_gen.get_workload(self.traffic_clock)
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

        # Cola fleet-wide: sumamos Lq (peticiones EN ESPERA, M/M/1) de todos los nodos activos.
        # Es el analogo simulado del qcur global del backend de HAProxy (cola compartida bajo leastconn).
        total_Lq = 0.0

        for i in range(self.n_max):
            idx = i * 6
            if self.sim_active_containers[i]:
                # M/M/1 -----------------------------------------------------------------------------------------
                
                # Params Base
                # mu: tasa de servicio POR NODO, constante y desacoplada de n_max (capacidad fisica fija).
                # Antes era total_users/n_max, lo que debilitaba cada nodo al aumentar n_max y dejaba la
                # flota completa exactamente en rho=1 durante el pico (cero margen -> sobreaprovisionamiento).
                max_capacity = NODE_CAPACITY
                base_latency_ms = 50.0 # S: Tiempo de servicio base sin hacer cola
                
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

                # Lq = peticiones en espera (largo de cola M/M/1) = L_total - rho (en servicio).
                # Lo acumulamos a la cola fleet-wide; el slot per-nodo se rellena tras el bucle.
                total_Lq += max(0.0, L_concurrent - rho)

                new_state[idx:idx+6] = [
                    min(1.0, max(0.0, cpu_usage)),             # cpu_usg
                    ram_usage,                                 # ram_usg_pct
                    0.0,                                       # queue_depth: placeholder, se rellena con la cola fleet-wide
                    min(1.0, latency_ms / 1000.0),             # latency normalizada a 1000ms (debe coincidir con MAX_LATENCY_MS de get_real_metrics)
                    min(1.0, max(0.0, errors)),                # error_rate
                    1.0                                        # status (ACTIVO)
                ]
            else:
                new_state[idx:idx+6] = [0.0] * 6 # Nodo apagado

        # Normalizamos la cola fleet-wide por (n_max * MAX_QUEUE_DEPTH): denominador fijo igual al real.
        # No varía con n_active_sim, de forma que queue_shared solo baja cuando la Lq real disminuye.
        n_active_sim = int(np.sum(self.sim_active_containers))
        if n_active_sim > 0:
            queue_shared = min(1.0, max(0.0, total_Lq / (self.n_max * MAX_QUEUE_DEPTH)))
            for i in range(self.n_max):
                if self.sim_active_containers[i]:
                    new_state[i * 6 + 2] = queue_shared

        # At the end, before returning new_state:
        workload_norm = np.float32(total_workload / self.traffic_gen.total_users)
        new_state = np.append(new_state, workload_norm)
        return new_state

    def reward_function(self, state, action, cant_active_containers):

            total_reward = 0.0

            W_LATENCY = self.reward_weights["W_LATENCY"]
            W_ERRORS = self.reward_weights["W_ERRORS"]
            W_COST = self.reward_weights["W_COST"]
            W_SATURATION = self.reward_weights["W_SATURATION"]
            W_OVERPROVISION = self.reward_weights["W_OVERPROVISION"]  # Penaliza nodos idle

            # Pesos para control de estabilidad
            W_SATURATION_PREVENTIVE = self.reward_weights["W_SATURATION_PREVENTIVE"]
            W_SCALE_FRICTION = self.reward_weights["W_SCALE_FRICTION"]  # Fricción para evitar el chattering (serrucho)
            W_QUEUE = self.reward_weights["W_QUEUE"]  # Backpressure: penaliza la profundidad de cola (señal anticipada de saturación)

            if cant_active_containers == 0:
                return -200.0

            # Banda objetivo de utilización POR NODO: dentro de [CPU_TARGET_MIN, CPU_TARGET_MAX]
            # un nodo opera de forma eficiente y no genera penalización.
            CPU_TARGET_MIN = 0.40
            CPU_TARGET_MAX = 0.85  # Frontera superior segura: permite operar los nodos más calientes (menos sobre-aprovisionamiento)
            CPU_SATURATION_HARD = 0.92  # Umbral de saturación extrema; deja una zona preventiva graduada [CPU_TARGET_MAX, CPU_SATURATION_HARD]

            # Variables para acumular
            total_latency_penalty_accum = 0.0
            avg_errors = 0.0
            avg_queue = 0.0
            total_ram_penalty_accum = 0.0
            total_cpu_penalty_accum = 0.0  # Unificamos todas las penalizaciones de CPU aquí
            
            for i in range(self.n_max):
                idx_base = i * 6 
                status = state[idx_base + 5] 
                
                if status == 1.0:
                    cpu_pct = state[idx_base]
                    ram_pct = state[idx_base + 1]
                    queue = state[idx_base + 2]
                    latency = state[idx_base + 3]
                    errores = state[idx_base + 4]

                    avg_errors += errores
                    avg_queue += queue
                    
                    # LOGICA DE LATENCIA POR NODO ---
                    node_latency_penalty = 0.0
                    
                    if latency > 0.1:
                        
                        node_latency_penalty = W_LATENCY * (latency ** 2)

                    total_latency_penalty_accum += node_latency_penalty

                    # LOGICA DE CPU POR NODO ---
                    node_cpu_penalty = 0.0
                    
                    if cpu_pct < CPU_TARGET_MIN:
                        # Castigo por ocioso
                        node_cpu_penalty = W_OVERPROVISION * (CPU_TARGET_MIN - cpu_pct)
                        
                    elif cpu_pct > CPU_TARGET_MAX:
                        # Castigo preventivo base (aplica a todo lo que pase de CPU_TARGET_MAX)
                        node_cpu_penalty = W_SATURATION_PREVENTIVE * (cpu_pct - CPU_TARGET_MAX)

                        # Si ADEMÁS pasa de CPU_SATURATION_HARD, le SUMAMOS el castigo de saturación extrema
                        # Esto asegura que la curva siempre suba y sea imposible hacer trampa
                        if cpu_pct > CPU_SATURATION_HARD:
                            node_cpu_penalty += W_SATURATION * (cpu_pct - CPU_SATURATION_HARD)
                            
                    total_cpu_penalty_accum += node_cpu_penalty
                    
        
                    if ram_pct > 0.85:
                        total_ram_penalty_accum += W_SATURATION * (ram_pct - 0.85)
                        
            # PROMEDIAMOS las métricas de los nodos encendidos
            latency_penalty_final = total_latency_penalty_accum / cant_active_containers
            avg_errors /= cant_active_containers
            avg_queue /= cant_active_containers
            cpu_penalty_final = total_cpu_penalty_accum / cant_active_containers
            ram_penalty_final = total_ram_penalty_accum / cant_active_containers
            

            # Calculo de penalizaciones
            error_penalty = W_ERRORS * avg_errors 
            cost_penalty = W_COST * (cant_active_containers / self.n_max)
            queue_penalty = W_QUEUE * avg_queue             

            total_reward -= (latency_penalty_final + error_penalty + cost_penalty + ram_penalty_final + queue_penalty + cpu_penalty_final)
            
            # Penalización por chattering: solo penaliza reversiones de dirección (up→down o down→up)
            scale_decision = action[-1]
            curr_dir = 1 if scale_decision >= 0.7 else (-1 if scale_decision <= 0.3 else 0)
            if (curr_dir != 0
                    and self.last_scale_direction != 0
                    and curr_dir != self.last_scale_direction):
                total_reward -= W_SCALE_FRICTION

            return float(total_reward)