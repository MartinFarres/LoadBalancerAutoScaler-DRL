import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LoadBalancerEnv(gym.Env):
    """
    Entorno PPO para Balanceo de Carga y Auto-scaling (LBDRL).
    """
    
    def __init__(self, n_max=10):
        super(LoadBalancerEnv, self).__init__()
        self.n_max = n_max
        
        # OBSERVATION SPACE
        # Por cada contenedor: [CPU, RAM, Tiempo_Respuesta, Tasa_Errores, Is_Active]
        # Todos los valores normalizados entre 0.0 y 1.0
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max * 5,), 
            dtype=np.float32
        )
        
        # ACTION SPACE
        # N_max valores para pesos de ruteo + 1 valor para decisión de auto-scaling. 
        # [w1, w2, ..., wN, decision_escalado]
        # Rango 0.0 a 1.0.
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_max + 1,), 
            dtype=np.float32
        )
        
        # Estado interno de la infraestructura simulada/real
        self.contenedores_activos = np.zeros(self.n_max, dtype=bool)
        self.estado_actual = np.zeros(self.n_max * 5, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reinicia el entorno, apagando el cluster y dejando 1 solo nodo activo."""
        super().reset(seed=seed)
        
        self.contenedores_activos = np.zeros(self.n_max, dtype=bool)
        self.contenedores_activos[0] = True # Nodo principal siempre vivo al inicio
        
        # TODO: Llamada al backend para hacer un reset real
        # ej: request.post("http://localhost:5000/api/cluster/reset")
        
        self.estado_actual = self.get_metrics()
        info = {"mensaje": "Cluster reiniciado a 1 instancia"}
        
        return self.estado_actual, info

    def step(self, action):
        """Avanza un instante de tiempo en el MDP."""
        # 1. Desempaquetar la accion generada por la red neuronal
        pesos_crudos = action[:self.n_max]
        decision_escalado = action[-1]
        
        # 2. Gestionar el Auto-scaling (Thresholds: <0.3 baja, >0.7 sube) <-- Estos valores pueden ser hiperparametros a ajustar
        if decision_escalado > 0.7:
            self.scale_up()
        elif decision_escalado < 0.3:
            self.scale_down()
            
        # 3. Aplicar Balanceo de Carga (Action Masking + Normalización)
        self.re_balance_traffick(pesos_crudos)
        
        # 4. Esperar / Procesar
        # Entorno real: time.sleep(2) o se espera el batch de requests
        # En simulacion: se calculan las colas matematicamente.
        
        # 5. Observar el impacto de nuestras acciones
        self.estado_actual = self.get_metrics()
        
        # 6. Calcular Recompensa
        recompensa = self.reward_function(self.estado_actual, action)
        
        # 7. Condiciones de término (todos los contenedores cayeron)
        terminated = not np.any(self.contenedores_activos)
        truncated = False # Se usa si definimos un MaxStepsPerEpisode
        
        info = {"activos": np.sum(self.contenedores_activos)}
        
        return self.estado_actual, recompensa, terminated, truncated, info

    def scale_up(self):
        """Encuentra el primer slot libre y levanta un contenedor."""
        inactivos = np.where(~self.contenedores_activos)[0]
        if len(inactivos) > 0:
            idx_nuevo = inactivos[0]
            self.contenedores_activos[idx_nuevo] = True
            # TODO: Llamada a Docker SDK para encender contenedor idx_nuevo
            # print(f"[Scaler] Levantando nodo {idx_nuevo}")

    def scale_down(self):
        """Apaga el último contenedor activo (protegiendo el nodo 0)."""
        activos = np.where(self.contenedores_activos)[0]
        if len(activos) > 1: # Prevenimos apagar todo el cluster
            idx_apagar = activos[-1]
            self.contenedores_activos[idx_apagar] = False
            # TODO: Llamada a Docker SDK para detener contenedor idx_apagar
            # print(f"[Scaler] Apagando nodo {idx_apagar}")

    def re_balance_traffick(self, pesos_crudos):
        """Enmascara nodos muertos, normaliza pesos y reconfigura el proxy."""
        # Aplicar máscara booleana (nodos apagados multiplican su peso por 0)
        pesos_filtrados = pesos_crudos * self.contenedores_activos
        
        suma_pesos = np.sum(pesos_filtrados)
        if suma_pesos > 0:
            pesos_finales = pesos_filtrados / suma_pesos
        else:
            # Failsafe de Sistemas Operativos: si la IA escupe todo 0 por error, 
            # hacemos un Fallback a Round Robin puro entre los nodos activos.
            pesos_finales = np.zeros(self.n_max)
            activos = np.where(self.contenedores_activos)[0]
            pesos_finales[activos] = 1.0 / len(activos)
            
        pesos_finales = np.round(pesos_finales, 4)
        
        # TODO: Actualizar pesos en Nginx via API o reescribiendo haproxy.cfg
        # print(f"[Proxy] Nuevos pesos enrutamiento: {pesos_finales}")

    def get_metrics(self):
        """Consulta el estado de la infraestructura y arma el vector de observaciones."""
        nuevo_estado = np.zeros(self.n_max * 5, dtype=np.float32)
        
        for i in range(self.n_max):
            if self.contenedores_activos[i]:
                # TODO: Aquí va la llamada al Docker SDK o API REST para leer métricas reales
                # Simulación de lectura de datos (valores dummy):
                cpu = np.random.uniform(0.1, 0.9)
                ram = np.random.uniform(0.2, 0.8)
                tiempo_respuesta = np.random.uniform(0.01, 0.15)
                errores = 0.0
                status = 1.0
                
                # Mapear las 5 métricas al vector plano
                idx_base = i * 5
                nuevo_estado[idx_base : idx_base+5] = [cpu, ram, tiempo_respuesta, errores, status]
            else:
                # Nodos apagados permanecen en ceros estrictos
                pass 
                
        return nuevo_estado

    def reward_function(self, estado, accion):
        """
        Calcula la puntuación del agente basándose en la latencia, 
        errores y uso eficiente de los contenedores.
        """
        pass