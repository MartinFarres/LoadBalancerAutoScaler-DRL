TOTAL_USERS = 45        # 45-90-135-180-225 Max concurrent Locust users; sets workload_norm scale in both sim and real env. Calibrado para ~1.25x headroom a full scale (n_max*NODE_CAPACITY / TOTAL_USERS), asi el pico necesita ~8 de 10 nodos.
NODE_CAPACITY = 10       # Sim M/M/1 per-node service rate mu (usuarios que un nodo sirve a rho=1). MEDIDO con utils/calibrate_node_capacity.py a 0.5 core / 4 gunicorn workers. Usamos 10 para que el agente sobre-aprovisione antes que saturar.
CONTAINER_CPU_CORES = 0.5  # CPU cores allocated per Docker container (nano_cpus = CONTAINER_CPU_CORES * 1e9). 25 nodos * 0.5 = 12.5 cores <= 16 del host (sin sobre-suscripcion/throttling).
USERS_PER_NODE = 9       # Usado SOLO por la busqueda robusta multi-escala (utils/sensitivity_analysis.py,
                          # Etapa 2): a cada tamano de flota en FLEET_SIZES se le asigna
                          # total_users = USERS_PER_NODE * n_max (45-90-135-180-225 para 5-10-15-20-25
                          # nodos), para que la relacion de headroom (n_max*NODE_CAPACITY/total_users)
                          # sea igual en todos los tamanos y la comparacion entre escalas sea justa. El
                          # resto del pipeline (Phase 1/2, tests, baselines) sigue usando TOTAL_USERS
                          # fijo de arriba, editado a mano por corrida segun el --nodes que corresponda.
MAX_MEMORY = 512         # MB of RAM per Docker container (Docker mem_limit AND the RAM normalization denominator)
MAX_QUEUE_DEPTH = 3.0     # Queue-depth NORMALIZATION ceiling: maps HAProxy qcur / sim Little's-law L into [0.0, 1.0]
SERVER_MAXCONN = 8         # HAProxy per-server connection limit ~ 2x los gunicorn workers (4). Bajo a proposito: HAProxy encola el excedente en qcur (la señal de backpressure que observa el agente) apenas el nodo se satura, y mantiene la cola consistente entre sim y real. Subirlo mucho oculta la saturacion dentro de gunicorn (qcur queda en 0).
SEED = 42                   # Global reproducibility seed for training, testing, and traffic generation

# --- Reward-shaping weights (ver LoadBalancerEnv.reward_function en environment/environment.py) ---
# Fuente unica de verdad de los pesos de recompensa. LoadBalancerEnv acepta un dict opcional
# reward_weights en el constructor que sobreescribe estos defaults. Este dict tambien define el
# espacio de muestreo para el analisis de sensibilidad Saltelli/Sobol en utils/sensitivity_analysis.py
# (el orden de las keys define el orden de variables del "problem" de SALib).
REWARD_WEIGHTS = {
    "W_LATENCY": 10.0,
    "W_ERRORS": 50.0,
    "W_COST": 5.0,
    "W_SATURATION": 15.0,
    "W_OVERPROVISION": 4.0,
    "W_SATURATION_PREVENTIVE": 5.0,
    "W_SCALE_FRICTION": 1.0,
    "W_QUEUE": 15.0,
}

# Rango [min, max] por peso para el muestreo de Saltelli (ver SALTELLI_SENSITIVITY.md para el
# razonamiento completo). Aproximadamente +-40-65% del default, salvo W_SCALE_FRICTION que se
# amplia mas: el README ya senala que el termino de anti-chattering "no cumple del todo su
# funcion" en la practica, asi que interesa ver si un valor mucho mayor lo corrige.
REWARD_WEIGHT_BOUNDS = {
    "W_LATENCY": [5.0, 20.0],
    "W_ERRORS": [25.0, 75.0],
    "W_COST": [2.5, 10.0],
    "W_SATURATION": [7.5, 25.0],
    "W_OVERPROVISION": [2.0, 8.0],
    "W_SATURATION_PREVENTIVE": [2.5, 10.0],
    "W_SCALE_FRICTION": [0.2, 3.0],
    "W_QUEUE": [7.5, 25.0],
}
