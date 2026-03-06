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
            self.relative_time = 0
            
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

        # Tope de seguridad: Nunca menos de 10 usuarios, nunca más del total_users
        user_count = max(10, min(user_count, self.total_users))

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
        mid_time = self.time_limit / 2

        if (self.relative_time <= mid_time):
            # Subida
            user_count = self.relative_time * self.scaling_rate + self.user_base
        else:
            # Bajada 
            # Calculamos pico medio
            peak_users = self.user_base + (self.scaling_rate * mid_time)
            # Tiempo bajando
            time_down = self.relative_time - mid_time
            # Restamos desde el pico
            user_count = peak_users - (self.scaling_rate * time_down)
            
        return user_count
    def exponential(self):
        # y = a * e^(k*t)
        # 70% del tiempo subida, 30% caida
        peak_time = self.time_limit * 0.7
        base_users = 50 
        
        if self.relative_time <= peak_time:
            # Subida 

            # Calculamos 'k' para que en el segundo final (time_limit) alcancemos el total_users
            # k = ln(total / base) / time_limit
            k = math.log(self.total_users / base_users) / max(1, peak_time)

            try:
                user_count = base_users * math.exp(k * self.relative_time)
            except OverflowError:
                user_count = self.total_users
        else:
            # Bajada
            time_down = self.relative_time - peak_time
            remaining_time = self.time_limit - peak_time
            drop_rate = (self.total_users - base_users) / max(1, remaining_time)
            user_count = self.total_users - (drop_rate * time_down)
            
        return user_count
    
    def step(self):
        mid_time = self.time_limit / 2

        # Calculamos dinamicamente el tamaño de los escalones
        step_duration_seconds = max(15, self.time_limit // 10) # 10 escalones en total aprox
        users_per_step = (self.total_users - self.user_base) / 5 # Sube en 5 escalones fuertes
        
        if self.relative_time <= mid_time:
            # subida
            current_step_index = self.relative_time // step_duration_seconds
            user_count = self.user_base + (users_per_step * current_step_index)
        else:
            # bajada

            # cuantos escalones logramos subir
            peak_steps = mid_time // step_duration_seconds
            peak_users = self.user_base + (users_per_step * peak_steps)
            
            # cuantos escalones debemos bajar
            time_down = self.relative_time - mid_time
            steps_down_index = time_down // step_duration_seconds
            user_count = peak_users - (users_per_step * steps_down_index)

        return user_count
