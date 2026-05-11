import subprocess
import time
import requests
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Load Balancer AutoScaler DRL run script')
    parser.add_argument('-p', '--pipeline', type=str, choices=['real', 'simulado', 'test_ppo', 'test_baseline', 'test_pid', 'all', 'sweep'], default='all', help='Pipeline to run')
    parser.add_argument('-n', '--nodes', type=int, default=5, help='Amount of nodes to run the training')
    parser.add_argument('-f', '--file', type=str, default='training_metrics.csv', help='Name of the CSV file to save metrics (will append timestamp to avoid overwrites)')
    parser.add_argument('-si', '--simulated_iterations', type=int, default=200000, help='Simulated training iterations')
    parser.add_argument('-ri', '--real_iterations', type=int, default=5000, help='Real training iterations')
    parser.add_argument('-ti', '--testing_iterations', type=int, default=1000, help='Testing iterations')
    parser.add_argument('-pi', '--pid_iterations', type=int, default=1000, help='PID baseline iterations')
    parser.add_argument('-ai', '--agent_iterations', type=int, default=1000, help='Agent baseline iterations')
    
    return parser.parse_args()

def simulated_training(args):
    processes = []

    print("[1/1] Iniciando Entrenamiento Simulado... ")

    cmd = [
        sys.executable, "environment/train_agent.py", "train_phase_1_simulation",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(args.simulated_iterations)
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


def real_training(processes, args):
    print("[1/1] Iniciando Entrenamiento Real... ")
    print("-" * 50)
    time.sleep(2)
    
    cmd = [
        sys.executable, "environment/train_agent.py", "train_phase_2_real_world",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(args.real_iterations)
    ]
    train_process = subprocess.Popen(cmd)
    processes.append(("PPO Training", train_process))

    try:
        train_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")

    down_processes(processes)


def test_agent(processes, args):
    print("[1/1] Iniciando Testing Agente PPO...")
    print("-" * 50)
    time.sleep(2)
   
    cmd = [
        sys.executable, "environment/test_agent.py",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(args.testing_iterations)
    ]
    test_process = subprocess.Popen(cmd)
    processes.append(("PPO Testing", test_process))

    try:
        test_process.wait()
    except KeyboardInterrupt:
        print("\n\n Testing interrumpido por el usuario (Ctrl+C).")
    
    down_processes(processes)


def test_baseline(processes, args):
    print("[1/1] Iniciando Testing Baseline (Umbrales Clásicos y Round Robin)...")
    print("-" * 50)
    time.sleep(2)
   
    cmd = [
        sys.executable, "environment/baseline_agent.py",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(args.agent_iterations)
    ]
    test_process = subprocess.Popen(cmd)
    processes.append(("Baseline Industry Testing", test_process))

    try:
        test_process.wait()
    except KeyboardInterrupt:
        print("\n\n Testing interrumpido por el usuario (Ctrl+C).")
    
    down_processes(processes)


def test_pid(processes, args):
    print("[1/1] Iniciando Testing Baseline (Controlador PID)...")
    print("-" * 50)
    time.sleep(2)
   
    cmd = [
        sys.executable, "environment/baseline_PID.py",
        "--nodes", str(args.nodes),
        "--file", args.file,
        "--iterations", str(args.pid_iterations)
    ]
    test_process = subprocess.Popen(cmd)
    processes.append(("Baseline PID Testing", test_process))

    try:
        test_process.wait()
    except KeyboardInterrupt:
        print("\n\n Testing interrumpido por el usuario (Ctrl+C).")
    
    down_processes(processes)

def wandb_sweep(args):
    print("[1/1] Lanzando proceso de W&B Sweep... ")
    
    cmd = [
        sys.executable, "environment/train_agent.py", "sweep",
        "--nodes", str(args.nodes),
        "--iterations", "200000" # Iteraciones recomendadas para el sweep
    ]
    sweep_process = subprocess.Popen(cmd)

    try:
        sweep_process.wait()
    except KeyboardInterrupt:
        print("\n\n Sweep interrumpido por el usuario (Ctrl+C).")


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
        res = requests.post(f"http://127.0.0.1:8000/init?n_max={nodes}", timeout=60)        
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
    for nombre, proceso in processes:
        print(f"   Deteniendo {nombre}...")
        proceso.terminate()
        proceso.wait() # Aseguramos que muera completamente
    
    try:
        requests.get("http://127.0.0.1:8000/cleanup", timeout=30)
    except:
        pass
        
    print("Terminado.")

if __name__ == "__main__":
    args = parse_args()
    comando = args.pipeline

    if comando == "simulado":
        simulated_training(args)
    elif comando == "real":
        real_training(init_processes(args.nodes), args) 
    elif comando == "test_ppo":
        test_agent(init_processes(args.nodes), args) 
    elif comando == "test_baseline":
        test_baseline(init_processes(args.nodes), args) 
    elif comando == "test_pid":
        test_pid(init_processes(args.nodes), args) 
    elif comando == "all":
        print("Iniciando pipeline completo (Simulación -> Real -> Testing Múltiple)...")
        simulated_training(args)
        real_training(init_processes(args.nodes), args) 
        
        print("\n\n" + "="*50)
        print("INICIANDO BATERÍA DE PRUEBAS COMPARATIVAS")
        print("="*50)
        
        test_baseline(init_processes(args.nodes), args) 
        test_pid(init_processes(args.nodes), args) 
        test_agent(init_processes(args.nodes), args) 
    elif comando == "sweep":
        wandb_sweep(args)
    else:
        print(f"Comando desconocido: {comando}")