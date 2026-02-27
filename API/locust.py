from locust import HttpUser, task, between

class MyStressUser(HttpUser):
    wait_time = between(1, 3) # Espera entre 1 y 3 segundos entre peticiones

    @task
    def hit_endpoint(self):
        # proxy expone la app en el puerto 80 local
        self.client.get("/")