from locust import HttpUser, task, between, LoadTestShape
import numpy as np
import math

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


class StressGenerator(LoadTestShape):
    """
    A class that generates a random number to pick a different function for trafficc generator.

    Settings:
        isOn --> check variable to let the stress continue or not
        total_users --> total users available for any given test
        function_number --> index for function to run
        time_limit --> total length for a unique test in ms
    """
    total_users = 40000 
    spawn_rate = 250 # Cuidado, aumentar puede colapsar el cpu
    running_fn = False
    function_tick_start = 0

    def tick(self):
        self.run_time = round(self.get_run_time())
        self.relative_time = self.run_time - self.function_tick_start

        if not self.running_fn or (self.relative_time >= self.time_limit):
            self.running_fn = True

            # Decidimos los tiempos limites y funciones a usar
            self.time_limit = np.random.randint(120, 900) # 2 mins to 15 mins
            self.function_number = np.random.randint(0,4)
            self.function_tick_start = self.run_time
            
            # Se generan los valores necesarios para cada funcion
            # Se generan ahora para prevenir que se generen por cada tick
            self.peak_one_users = self.total_users * np.random.uniform(0.5, 0.9)
            self.min_users = self.total_users * np.random.uniform(0.05, 0.20)
            self.peak_two_users = self.total_users * np.random.uniform(0.5, 0.9)
            self.user_base = self.total_users * np.random.uniform(0.02, 0.15)
            self.scaling_rate = np.random.randint(1, 10)
            
        # llamar a funcion y devolver resultado
        match self.function_number:
            case 0:
                user_count = self.doubleWave()
            case 1: 
                user_count = self.lineal()
            case 2: 
                user_count = self.exponential()
            case 3:
                user_count = self.step()

        return user_count, self.spawn_rate


    def doubleWave(self):
        user_count = (
                (self.peak_one_users - self.min_users)
                * math.e ** -(((self.relative_time / (self.time_limit / 10 * 2 / 3)) - 5) ** 2)
                + (self.peak_two_users - self.min_users)
                * math.e ** -(((self.relative_time / (self.time_limit / 10 * 2 / 3)) - 10) ** 2)
                + self.min_users
            )
        return (round(user_count))


    def lineal(self):
        # y = mx + c
        user_count = self.relative_time * self.scaling_rate + self.user_base
        return (round(user_count))

    def exponential(self):
        # y = a * e^(k*t)
        base_users = 50 
        
        # Calculamos 'k' para que en el segundo final (time_limit) alcancemos el total_users
        # k = ln(total / base) / time_limit
        k = math.log(self.total_users / base_users) / max(1, self.time_limit)
        
        try:
            user_count = base_users * math.exp(k * self.relative_time)
        except OverflowError:
            user_count = self.total_users
            
        # Tope de seguridad
        user_count = min(user_count, self.total_users)
        
        return (round(user_count))
    
    def step(self):
        # Configuramos los escalones
        step_duration_seconds = 60 # Cada 1 minuto damos un salto
        users_per_step = self.total_users * 0.10 # Saltos del 10% del total
        base_users = 100
        
        # La división entera
        current_step_index = self.relative_time // step_duration_seconds
        
        user_count = base_users + (users_per_step * current_step_index)
        
        # Tope de seguridad
        user_count = min(user_count, self.total_users)
        
        return (round(user_count))
