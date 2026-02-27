from locust import HttpUser, task, between

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