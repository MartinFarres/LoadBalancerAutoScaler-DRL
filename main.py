import subprocess
import time
import requests
import sys
import argparse


# Iteraciones por defecto de cada pipeline cuando no se pasa --iterations.
# 'all' es multi-fase, por eso usa las claves individuales de aqui en vez de un solo valor.
DEFAULT_ITERS = {
    'simulado': 200000,
    'real': 5000,
    'tests_sim': 1000,
    'tests_real': 1000,
    'test_ppo': 1000,
    'test_baseline': 1000,
    'test_pid': 1000,
    'sweep': 200000,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Load Balancer AutoScaler DRL run script')
    parser.add_argument('-p', '--pipeline', type=str,
                        choices=['real', 'simulado', 'tests_sim', 'tests_real',
                                 'test_ppo', 'test_baseline', 'test_pid', 'all', 'sweep'],
                        default='all', help='Pipeline to run')
    parser.add_argument('-n', '--nodes', type=int, default=5, help='Amount of nodes to run the training')
    parser.add_argument('-f', '--file', type=str, default='training_metrics.csv', help='Name of the CSV file to save metrics (will append timestamp to avoid overwrites)')
    parser.add_argument('-i', '--iterations', type=int, default=None,
                        help='Unified iteration/timestep count for the selected pipeline. Used when the '
                             'pipeline-specific flag below is not given; otherwise falls back to a per-pipeline '
                             'default. Does not apply to "all" (multi-phase) — use the specific flags there.')

    # Flags especificos por pipeline. Precedencia: especifico > -i/--iterations > DEFAULT_ITERS.
    # Default None para que la precedencia funcione (un valor no-None se interpreta como "set").
    parser.add_argument('-si', '--simulated_iterations', type=int, default=None, help='Timesteps for Phase 1 simulated training (simulado / all). Overrides -i.')
    parser.add_argument('-ri', '--real_iterations', type=int, default=None, help='Timesteps for Phase 2 real-world fine-tuning (real / all). Overrides -i.')
    parser.add_argument('-ti', '--testing_iterations', type=int, default=None, help='Steps for the PPO agent test (test_ppo). Overrides -i.')
    parser.add_argument('-pi', '--pid_iterations', type=int, default=None, help='Steps for the PID baseline test (test_pid). Overrides -i.')
    parser.add_argument('-ai', '--agent_iterations', type=int, default=None, help='Steps for the industry baseline test (test_baseline). Overrides -i.')
    parser.add_argument('-sti', '--sim_test_iterations', type=int, default=None, help='Steps for the simulated test battery (tests_sim / all sim tests). Overrides -i.')
    parser.add_argument('-rti', '--real_test_iterations', type=int, default=None, help='Steps for the real test battery (tests_real / all real tests). Overrides -i.')

    return parser.parse_args()


# Mapea cada pipeline a su flag especifico de iteraciones (nombre del atributo en args).
SPECIFIC_ATTR = {
    'simulado': 'simulated_iterations',
    'real': 'real_iterations',
    'tests_sim': 'sim_test_iterations',
    'tests_real': 'real_test_iterations',
    'test_ppo': 'testing_iterations',
    'test_baseline': 'agent_iterations',
    'test_pid': 'pid_iterations',
}


def resolve_iters(args, pipeline, use_global=True):
    """Precedencia: flag especifico del pipeline > -i/--iterations > default del pipeline.
    use_global=False ignora -i (lo usa 'all', que es multi-fase y no puede mapear un solo -i)."""
    attr = SPECIFIC_ATTR.get(pipeline)
    if attr is not None:
        specific = getattr(args, attr, None)
        if specific is not None:
            return specific
    if use_global and args.iterations is not None:
        return args.iterations
    return DEFAULT_ITERS[pipeline]

def simulated_training(args, iterations):
    processes = []

    print("[1/1] Iniciando Entrenamiento Simulado... ")

    cmd = [
        sys.executable, "environment/train_agent.py", "train_phase_1_simulation",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(iterations)
    ]
    train_process = subprocess.Popen(cmd)
    processes.append(("PPO Training Simulated", train_process))

    try:
        # esperando a que el entrenamiento termine (o a que presiones Ctrl+C)
        train_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")
        sys.exit(0) # Salida limpia
    finally:
        pass
        # down_processes(processes)


def real_training(processes, args, iterations):
    print("[1/1] Iniciando Entrenamiento Real... ")
    print("-" * 50)
    time.sleep(2)

    cmd = [
        sys.executable, "environment/train_agent.py", "train_phase_2_real_world",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(iterations)
    ]
    train_process = subprocess.Popen(cmd)
    processes.append(("PPO Training", train_process))

    try:
        train_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")

    down_processes(processes)


def _run_test(script, name, extra_args, simulated):
    """Start a test subprocess, wait for it, return (process_name, process) tuple."""
    cmd = [sys.executable, script] + extra_args
    if simulated:
        cmd.append("--simulated")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n\n Testing interrumpido por el usuario (Ctrl+C).")
    return (name, proc)


def test_agent(args, simulated=False, iterations=1000):
    iters = iterations
    mode = "Simulado" if simulated else "Real"
    print(f"[1/1] Iniciando Testing Agente PPO ({mode})...")
    print("-" * 50)

    processes = []
    if not simulated:
        time.sleep(2)
        processes = init_processes(args.nodes)

    processes.append(_run_test(
        "environment/test_agent.py", "PPO Testing",
        ["--nodes", str(args.nodes), "--file", args.file, "--iterations", str(iters)],
        simulated
    ))

    if not simulated:
        down_processes(processes)


def test_baseline(args, simulated=False, iterations=1000):
    iters = iterations
    mode = "Simulado" if simulated else "Real"
    print(f"[1/1] Iniciando Testing Baseline BAI ({mode})...")
    print("-" * 50)

    processes = []
    if not simulated:
        time.sleep(2)
        processes = init_processes(args.nodes)

    processes.append(_run_test(
        "environment/baseline_agent.py", "Baseline Industry Testing",
        ["--nodes", str(args.nodes), "--file", args.file, "--iterations", str(iters)],
        simulated
    ))

    if not simulated:
        down_processes(processes)


def test_pid(args, simulated=False, iterations=1000):
    iters = iterations
    mode = "Simulado" if simulated else "Real"
    print(f"[1/1] Iniciando Testing Baseline PID ({mode})...")
    print("-" * 50)

    processes = []
    if not simulated:
        time.sleep(2)
        processes = init_processes(args.nodes)

    processes.append(_run_test(
        "environment/baseline_PID.py", "Baseline PID Testing",
        ["--nodes", str(args.nodes), "--file", args.file, "--iterations", str(iters)],
        simulated
    ))

    if not simulated:
        down_processes(processes)

def wandb_sweep(args, iterations):
    print("[1/1] Lanzando proceso de W&B Sweep... ")

    cmd = [
        sys.executable, "environment/train_agent.py", "sweep",
        "--nodes", str(args.nodes),
        "--iterations", str(iterations)
    ]
    sweep_process = subprocess.Popen(cmd)

    try:
        sweep_process.wait()
    except KeyboardInterrupt:
        print("\n\n Sweep interrumpido por el usuario (Ctrl+C).")


def run_sim_tests(args, iterations):
    """Corre las tres pruebas comparativas (BAI, PID, PPO) en entorno SIMULADO."""
    print("\n--- Tests en entorno SIMULADO ---")
    test_baseline(args, simulated=True, iterations=iterations)
    test_pid(args, simulated=True, iterations=iterations)
    test_agent(args, simulated=True, iterations=iterations)


def run_real_tests(args, iterations):
    """Corre las tres pruebas comparativas (BAI, PID, PPO) en entorno REAL.
    Cada prueba levanta y apaga su propio cluster (igual que el comportamiento previo)."""
    print("\n--- Tests en entorno REAL ---")
    test_baseline(args, simulated=False, iterations=iterations)
    test_pid(args, simulated=False, iterations=iterations)
    test_agent(args, simulated=False, iterations=iterations)


def init_processes(nodes):
    processes = []

    print(" Iniciando Entorno Real... ")

    print("[1/4] Iniciando API Bridge (FastAPI)...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "bridge:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"], 
        cwd="API"
    )
    processes.append(("API Bridge", api_process))
    
    time.sleep(3) 

    print("[2/4] Inicializando cluster Docker (HAProxy + Nodos)...")

    try:
        # Hacemos el POST al init. Le damos un timeout largo porque Docker tiene que crear contenedores
        res = requests.post(f"http://127.0.0.1:8000/init?n_max={nodes}", timeout=120)        
        if res.status_code == 200:
            print("  Cluster inicializado con éxito.")
        else:
            print(f"  Error al inicializar: {res.text}")
            raise Exception("API devolvió error")
    except Exception as e:
        print("   Fallo la conexión con la API:", e)
        down_processes(processes)
        sys.exit(1)

    # Esperamos a que los contenedores respiren y HAProxy resuelva los DNS internos
    time.sleep(5) 

    print("[3/4] Iniciando tráfico de estrés (Locust Headless)...")
    # -u 50: 50 usuarios concurrentes | -r 1: entran 1 por segundo
    locust_process = subprocess.Popen([
        sys.executable, "-m", "locust", "-f", "API/locust.py", 
        "--headless",  
        "-H", "http://127.0.0.1:80"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("Locust", locust_process))

    print("Tráfico simulado inyectándose en http://127.0.0.1:80")

    print("[4/4] Iniciando TensorBoard...")

    tb_process = subprocess.Popen([sys.executable, "-m", "tensorboard", "--logdir", "./logs_tensorboard/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("TensorBoard", tb_process))

    print("    TensorBoard disponible en http://localhost:6006")

    return processes


def down_processes(processes):
    print("\n [Limpieza] Apagando servicios en segundo plano...")
    
    try:
        requests.post("http://127.0.0.1:8000/cleanup", timeout=30)
    except:
        pass

    for nombre, proceso in processes:
        print(f"   Deteniendo {nombre}...")
        proceso.terminate()
        proceso.wait() # Aseguramos que muera completamente
    
    time.sleep(5) # Tiempo para que docker termine de matar los procesos
    print("Terminado.")

if __name__ == "__main__":
    args = parse_args()
    comando = args.pipeline

    # 'all' es multi-fase y no puede mapear un unico --iterations a sus fases.
    if comando == "all" and args.iterations is not None:
        print("[Aviso] --iterations (-i) no aplica al pipeline 'all'. "
              "Use -si/-ri/-sti/-rti para ajustar cada fase; el resto usa defaults.")

    if comando == "simulado":
        simulated_training(args, resolve_iters(args, "simulado"))
    elif comando == "real":
        real_training(init_processes(args.nodes), args, resolve_iters(args, "real"))
    elif comando == "tests_sim":
        run_sim_tests(args, resolve_iters(args, "tests_sim"))
    elif comando == "tests_real":
        run_real_tests(args, resolve_iters(args, "tests_real"))
    elif comando == "test_ppo":
        iters = resolve_iters(args, "test_ppo")
        test_agent(args, simulated=True, iterations=iters)
        test_agent(args, simulated=False, iterations=iters)
    elif comando == "test_baseline":
        iters = resolve_iters(args, "test_baseline")
        test_baseline(args, simulated=True, iterations=iters)
        test_baseline(args, simulated=False, iterations=iters)
    elif comando == "test_pid":
        iters = resolve_iters(args, "test_pid")
        test_pid(args, simulated=True, iterations=iters)
        test_pid(args, simulated=False, iterations=iters)
    elif comando == "all":
        print("Iniciando pipeline completo (Simulación -> Real -> Testing Múltiple)...")
        simulated_training(args, resolve_iters(args, "simulado", use_global=False))
        real_training(init_processes(args.nodes), args, resolve_iters(args, "real", use_global=False))

        print("\n\n" + "="*50)
        print("INICIANDO BATERÍA DE PRUEBAS COMPARATIVAS")
        print("="*50)

        run_sim_tests(args, resolve_iters(args, "tests_sim", use_global=False))
        run_real_tests(args, resolve_iters(args, "tests_real", use_global=False))
    elif comando == "sweep":
        wandb_sweep(args, resolve_iters(args, "sweep"))
    else:
        print(f"Comando desconocido: {comando}")