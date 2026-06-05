import argparse
import os
import random
import statistics
import sys
import threading
import time

import docker
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CONTAINER_CPU_CORES, MAX_MEMORY

# Mezcla de endpoints identica a la de API/locust.py (pesos 6:3:1).
REQUEST_MIX = ["/"] * 6 + ["/cpu"] * 3 + ["/ram"] * 1
# Think-time por usuario, igual que locust wait_time = between(1, 3).
THINK_MIN, THINK_MAX = 1.0, 3.0
# Techo de latencia del entorno (MAX_LATENCY_MS en environment.py): por encima de esto el
# agente ve latencia normalizada = 1.0 (penalizacion maxima). Lo usamos como umbral de rodilla.
MAX_LATENCY_MS = 1000.0

CONTAINER_NAME = "lbas_calib_node"


def start_node(client, port, cpu_cores, mem_mb):
    """Levanta un unico dummy_server con los MISMOS limites que produccion y publica su puerto."""
    # Limpiamos cualquier sobra de una corrida previa.
    try:
        old = client.containers.get(CONTAINER_NAME)
        old.remove(force=True)
    except docker.errors.NotFound:
        pass

    try:
        client.images.get("dummy_server:latest")
    except docker.errors.ImageNotFound:
        print("ERROR: falta la imagen dummy_server:latest.")
        print("  Construila con: cd API/dummy_server && docker build -t dummy_server:latest .")
        sys.exit(1)

    container = client.containers.run(
        image="dummy_server:latest",
        name=CONTAINER_NAME,
        detach=True,
        nano_cpus=int(cpu_cores * 1_000_000_000),   # identico a ClusterOrchestration.start()
        mem_limit=f"{mem_mb}m",                       # identico a ClusterOrchestration.start()
        ports={"8000/tcp": port},
        labels={"role": "lbas_calib"},
    )
    return container


def wait_ready(url, timeout=30.0):
    """Espera a que el nodo responda 200 en '/' antes de empezar a medir."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url + "/", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_level(url, n_users, duration, warmup):
    """
    Simula n_users clientes en bucle cerrado (request -> think-time -> request) durante
    `duration` segundos (tras `warmup` segundos de descarte). Devuelve metricas agregadas.
    """
    stop_flag = threading.Event()
    measuring = threading.Event()
    latencies = []        # ms, solo de la ventana de medicion
    errors = [0]
    ok = [0]
    lock = threading.Lock()

    def user_loop():
        session = requests.Session()
        rng = random.Random()
        while not stop_flag.is_set():
            endpoint = rng.choice(REQUEST_MIX)
            t0 = time.perf_counter()
            try:
                r = session.get(url + endpoint, timeout=15)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if measuring.is_set():
                    with lock:
                        if r.status_code >= 500:
                            errors[0] += 1
                        else:
                            ok[0] += 1
                            latencies.append(dt_ms)
            except Exception:
                if measuring.is_set():
                    with lock:
                        errors[0] += 1
            # Think-time del usuario (igual que locust between(1,3)).
            time.sleep(rng.uniform(THINK_MIN, THINK_MAX))

    threads = [threading.Thread(target=user_loop, daemon=True) for _ in range(n_users)]
    for t in threads:
        t.start()

    time.sleep(warmup)          # descartamos el transitorio de arranque
    measuring.set()
    time.sleep(duration)        # ventana de medicion
    measuring.clear()
    stop_flag.set()
    for t in threads:
        t.join(timeout=16)

    with lock:
        lat = sorted(latencies)
        total = ok[0] + errors[0]
        return {
            "users": n_users,
            "throughput": ok[0] / duration,
            "p50": statistics.median(lat) if lat else float("nan"),
            "p95": lat[int(0.95 * (len(lat) - 1))] if lat else float("nan"),
            "p99": lat[int(0.99 * (len(lat) - 1))] if lat else float("nan"),
            "error_pct": (100.0 * errors[0] / total) if total else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Calibra NODE_CAPACITY midiendo un nodo real.")
    parser.add_argument("--levels", default="5,10,20,40,80,160,320,640",
                        help="Niveles de usuarios concurrentes, separados por coma.")
    parser.add_argument("--duration", type=float, default=15.0, help="Segundos de medicion por nivel.")
    parser.add_argument("--warmup", type=float, default=4.0, help="Segundos de calentamiento por nivel.")
    parser.add_argument("--port", type=int, default=8011, help="Puerto del host para el nodo de prueba.")
    parser.add_argument("--cpu-cores", type=float, default=CONTAINER_CPU_CORES,
                        help="Cores por contenedor (default: config.CONTAINER_CPU_CORES).")
    parser.add_argument("--mem", type=int, default=MAX_MEMORY,
                        help="MB de RAM por contenedor (default: config.MAX_MEMORY).")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    url = f"http://127.0.0.1:{args.port}"
    client = docker.from_env()

    print(f"Levantando nodo de prueba: {args.cpu_cores} cores, {args.mem} MB RAM "
          f"(mezcla 6:3:1, think-time {THINK_MIN}-{THINK_MAX}s, igual que locust.py)")
    container = start_node(client, args.port, args.cpu_cores, args.mem)
    try:
        if not wait_ready(url):
            print("ERROR: el nodo no respondio a tiempo.")
            return
        print("Nodo listo. Iniciando barrido de concurrencia...\n")

        header = f"{'users':>6} | {'throughput':>11} | {'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8} | {'err %':>6}"
        print(header)
        print("-" * len(header))

        results = []
        for n in levels:
            r = run_level(url, n, args.duration, args.warmup)
            results.append(r)
            print(f"{r['users']:>6} | {r['throughput']:>9.1f}/s | {r['p50']:>8.0f} | "
                  f"{r['p95']:>8.0f} | {r['p99']:>8.0f} | {r['error_pct']:>5.1f}%")

        # --- Analisis de la rodilla -------------------------------------------------------
        peak_tp = max(r["throughput"] for r in results)
        # Saturacion = primer nivel donde p95 cruza el techo de latencia del entorno (1000ms)
        # o donde aparecen errores 5xx. La capacidad util es el nivel JUSTO ANTES.
        knee_idx = None
        for i, r in enumerate(results):
            if (r["p95"] >= MAX_LATENCY_MS) or (r["error_pct"] > 1.0):
                knee_idx = i
                break

        print()
        if knee_idx is None:
            # Nunca se saturo: la capacidad es al menos el nivel mas alto probado.
            sustained = levels[-1]
            print(f"No se alcanzo la rodilla (p95 < {MAX_LATENCY_MS:.0f}ms en todos los niveles).")
            print(f"NODE_CAPACITY es >= {sustained} usuarios; ampliar --levels para encontrar el limite.")
        else:
            sustained = levels[knee_idx - 1] if knee_idx > 0 else levels[0]
            r_knee = results[knee_idx]
            print(f"Rodilla detectada en {levels[knee_idx]} usuarios "
                  f"(p95={r_knee['p95']:.0f}ms, err={r_knee['error_pct']:.1f}%).")
            print(f"Ultimo nivel sano: {sustained} usuarios.")

        print()
        print(f"Throughput pico medido: {peak_tp:.1f} req/s")
        print("=" * 60)
        print(f"  SUGERENCIA: NODE_CAPACITY ~= {sustained}")
        print("=" * 60)

    finally:
        print("\nLimpiando contenedor de prueba...")
        try:
            container.remove(force=True)
        except Exception as e:
            print(f"  (no se pudo remover {CONTAINER_NAME}: {e})")


if __name__ == "__main__":
    main()
