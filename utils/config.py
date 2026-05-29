TOTAL_USERS = 3000       # Max concurrent Locust users; sets workload_norm scale in both sim and real env
NODE_CAPACITY = 400      # Sim M/M/1 per-node service rate mu (users a single node serves at rho=1), fixed and independent of n_max. Fleet headroom = n_max*NODE_CAPACITY / TOTAL_USERS (e.g. 1.5x at n_max=5).
CONTAINER_CPU_CORES = 2.0  # CPU cores allocated per Docker container (nano_cpus = CONTAINER_CPU_CORES * 1e9)
MAX_MEMORY = 512         # MB of RAM per Docker container (Docker mem_limit AND the RAM normalization denominator)
MAX_QUEUE_DEPTH = 10.0     # Queue-depth NORMALIZATION ceiling: maps HAProxy qcur / sim Little's-law L into [0.0, 1.0]
SERVER_MAXCONN = 4         # HAProxy per-server connection limit ~ Gunicorn workers: cuándo HAProxy empieza a encolar (qcur>0). Mantener bajo (≈workers) para que la cola refleje saturación real.
SEED = 42                   # Global reproducibility seed for training, testing, and traffic generation
