from locust import HttpUser, task, between, LoadTestShape
import numpy as np
import math
import sys
import os
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.traffic_generator import TrafficGenerator

class StressUser(HttpUser):
    # Simula el tiempo que un usuario real se queda leyendo la pantalla antes de hacer otro click
    wait_time = between(1, 3)

    # Los numeros dentro de @task() son los "pesos".
    # Esto significa que el endpoint normal se visitara 6 veces mas que el de RAM.
    @task(6)
    def traffic_normal(self):
        """Simula navegación típica (Carga baja)"""
        self.client.get("/")

    @task(3)
    def traffic_cpu(self):
        """Simula una petición de procesamiento complejo (ej. exportar un PDF o buscar en BD)"""
        self.client.get("/cpu")

    @task(1)
    def traffic_ram(self):
        """Simula una subida de archivo grande o caché pesada"""
        self.client.get("/ram")

BRIDGE_URL = "http://127.0.0.1:8000"

class StressGenerator(LoadTestShape):
    """
    A class that generates a random number to pick a different function for traffic generator.

    Settings:
        isOn --> check variable to let the stress continue or not
        total_users --> total users available for any given test
        function_number --> index for function to run
        time_limit --> total length for a unique test in ms
    """
    total_users = 4000 
    spawn_rate = 40 # Cuidado, aumentar puede colapsar el cpu
    running_fn = False
    function_tick_start = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inicializamos el generador con los parámetros para el mundo real
        self.traffic_gen = TrafficGenerator(total_users=4000, min_duration=120, max_duration=900)

    def tick(self):
        run_time = self.get_run_time()

        # Obtenemos la carga pasándole el tiempo de Locust
        user_count = self.traffic_gen.get_workload(run_time)

        # Pusheamos a la api bridge
        try:
            requests.post(
                f"{BRIDGE_URL}/workload",
                params={"user_count": round(user_count), "total_users": self.total_users},
                timeout=0.2  # non-blocking, don't stall Locust
            )
        except Exception:
            pass
        
        return round(user_count), self.spawn_rate