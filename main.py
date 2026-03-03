import subprocess
import time
import requests
import sys

def simulated_training():
    processes = []

    print("[1/1] Iniciando Entrenamiento Simulado... ")

    train_process = subprocess.Popen(["python", "environment/train_agent.py", "train_phase_1_simulation"])
    processes.append(("PPO Training Simulated", train_process))

    try:
        # esperando a que el entrenamiento termine (o a que presiones Ctrl+C)
        train_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")
        sys.exit(0) # Salida limpia
    finally:
        down_processes(processes)


def real_training(processes):

    print("[1/1] Iniciando Entrenamiento Real... ")
    print("-" * 50)
    time.sleep(2)
    
    train_process = subprocess.Popen(["python", "environment/train_agent.py", "train_phase_2_real_world"])
    processes.append(("PPO Training", train_process))

    try:
        # esperando a que el entrenamiento termine (o a que presiones Ctrl+C)
        train_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")

    down_processes(processes)

def test_agent(processes):

    print("[1/1] Iniciando Testing Agente...")
    print("-" * 50)
    time.sleep(2)
   
    test_process = subprocess.Popen(["python", "environment/test_agent.py"])
    processes.append(("PPO Testing", test_process))

    try:
        test_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")
    
    down_processes(processes)


def init_processes():
    processes = []

    print(" Iniciando Entorno Real... ")

    print("[1/4] Iniciando API Bridge (FastAPI)...")
    api_process = subprocess.Popen(
        ["uvicorn", "bridge:app", "--host", "0.0.0.0", "--port", "8000"], 
        cwd="API"
    )
    processes.append(("API Bridge", api_process))
    
    time.sleep(3) 

    print("[2/4] Inicializando cluster Docker (HAProxy + Nodos)...")

    try:
        # Hacemos el POST al init. Le damos un timeout largo porque Docker tiene que crear contenedores
        res = requests.post("http://127.0.0.1:8000/init", timeout=60)
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
        "locust", "-f", "API/locust.py", 
        "--headless", "-u", "50", "-r", "1", 
        "-H", "http://127.0.0.1:80"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("Locust", locust_process))

    print("Tráfico simulado inyectándose en http://127.0.0.1:80")

    print("[4/4] Iniciando TensorBoard...")

    tb_process = subprocess.Popen(["tensorboard", "--logdir", "./logs_tensorboard/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    if len(sys.argv) > 1:
        comando = sys.argv[1]
        
        if comando == "simulado":
            simulated_training()
        elif comando == "real":
            real_training(init_processes())
        elif comando == "test":
            test_agent(init_processes())
        else:
            print(f"Comando desconocido: {comando}")
    else:
        print("Iniciando pipeline completo (Simulación -> Real -> Testing )...")
        simulated_training()
        real_training(init_processes())
        test_agent(init_processes())