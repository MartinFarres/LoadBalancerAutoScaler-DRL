import subprocess
import time
import requests
import sys

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", action="store_true", help="Corre todo en modo matemático (sin Docker)")
    args = parser.parse_args()

    procesos = []
    is_simulated = args.simulated

    if not is_simulated:
        print("[1/5] Iniciando API Bridge (FastAPI)...")
        api_process = subprocess.Popen(
            ["uvicorn", "bridge:app", "--host", "0.0.0.0", "--port", "8000"], 
            cwd="API"
        )
        procesos.append(("API Bridge", api_process))
        
        time.sleep(3) 

        print("[2/5] Inicializando cluster Docker (HAProxy + Nodos)...")
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
            apagar_procesos(procesos)
            sys.exit(1)

        # Esperamos a que los contenedores respiren y HAProxy resuelva los DNS internos
        time.sleep(5) 

        print("[3/5] Iniciando tráfico de estrés (Locust Headless)...")
        # -u 50: 50 usuarios concurrentes | -r 5: entran 5 por segundo
        locust_process = subprocess.Popen([
            "locust", "-f", "API/locust.py", 
            "--headless", "-u", "50", "-r", "5", 
            "-H", "http://127.0.0.1:80"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procesos.append(("Locust", locust_process))
        print("   Tráfico simulado inyectándose en http://127.0.0.1:80")
    else:
        print("--- [MODO SIMULADO ACTIVADO] ---")
        print("Saltando inicialización de API Bridge, Docker y Locust.")

    print("[4/5] Iniciando TensorBoard...")
    tb_process = subprocess.Popen(["tensorboard", "--logdir", "./logs_tensorboard/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procesos.append(("TensorBoard", tb_process))
    print("    TensorBoard disponible en http://localhost:6006")

    print(" [5/5] Iniciando Entrenamiento del Agente PPO...\n")
    print("-" * 50)
    time.sleep(2)
    
    # Pasamos el flag al proceso de entrenamiento
    train_cmd = ["python", "environment/train_agent.py"]
    if is_simulated:
        train_cmd.append("--simulated")

    train_process = subprocess.Popen(train_cmd)
    procesos.append(("PPO Training", train_process))

    try:
        # esperando a que el entrenamiento termine (o a que presiones Ctrl+C)
        train_process.wait()
    except KeyboardInterrupt:
        print("\n\n Entrenamiento interrumpido por el usuario (Ctrl+C).")

    apagar_procesos(procesos)

def apagar_procesos(procesos):
    print("\n [Limpieza] Apagando servicios en segundo plano...")
    for nombre, proceso in procesos:
        print(f"   Deteniendo {nombre}...")
        proceso.terminate()
        proceso.wait() # Aseguramos que muera completamente
    
    try:
        requests.get("http://127.0.0.1:8000/cleanup", timeout=30)
    except:
        pass
        
    print("Terminado.")

if __name__ == "__main__":
    main()