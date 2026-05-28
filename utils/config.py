TOTAL_USERS = 1500       # Max concurrent Locust users; sets workload_norm scale in both sim and real env
CONTAINER_CPU_CORES = 1.0  # CPU cores allocated per Docker container (nano_cpus = CONTAINER_CPU_CORES * 1e9)
MAX_MEMORY = 1024          # MB of RAM per Docker container (Docker mem_limit AND the RAM normalization denominator)
MAX_QUEUE_DEPTH = 50.0     # Queue-depth normalization ceiling: maps HAProxy qcur / sim Little's-law L into [0.0, 1.0]
SEED = 42                   # Global reproducibility seed for training, testing, and traffic generation
