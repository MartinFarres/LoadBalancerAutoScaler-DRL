import docker
import socket
import csv
import io
from bridge import ContainerMetrics


class ClusterOrchestration():
    def __init__(self):
        pass

    def set_params_and_start(self, n_max=10, max_memory=1024, node_name="lbas_node"):
        self.n_max = n_max
        self.node_name = node_name
        self.last_active_container_idx = 0
        self.max_memory = max_memory
        
        self.client = docker.from_env()
        # Pull image or Create image in dockerfile with a functioning server
        self.image_container = self.client.images.get("My Image")
        # HAProxy image
        self.image_HAProxy = self.client.images.get("haproxytech/haproxy-alpine:3.0")

        self.start()

    def start(self):
        # Create network if doesn't exists
        try:
            self.client.networks.get("lbas_network")
            # Stop all containers
            self.stop_all()
        except docker.errors.NotFound:
            self.client.networks.create("lbas_network")

        # Creates all n containers
        for i in range(self.n_max):
            self.client.containers.run(image=self.image_container, network="lbas_network", detach=True, name=f"{self.node_name}_{i}",labels={"role": "lbas_node"})

        # Creates HAProxy container and its configuration
        self.init_haproxy_cfg()
        self.client.containers.run(image=self.image_HAProxy, 
                                                network="lbas_network", 
                                                name=f"lbas_haproxy", 
                                                volumes={
                                                    # path on your machine/host
                                                    "./haproxy.cfg": {
                                                        "bind": "/usr/local/etc/haproxy/haproxy.cfg",  # path inside the container
                                                        "mode": "rw",
                                                    },
                                                },
                                                ports={'80/tcp': 80, '9999/tcp': 9999},
                                                detach=True,
                                                labels={"role":"lbas_haproxy"})
        
        
        
    def stop_all(self):
        for container in self.client.containers.list(filters={"label": "role=lbas_node"}):
            container.stop()
        try:
            self.client.containers.get("lbas_haproxy").stop()
            self.client.containers.get("lbas_haproxy").remove()
        except docker.errors.NotFound:
            pass

    def reset(self):
        self.last_active_container_idx = 0
        # Apagamos el trafico para todos menos el primero
        for i in range(1, self.n_max):
            self.send_haproxy_command(f"set weight servidores_web/{self.node_name}_{i} 0")
        self.send_haproxy_command(f"set weight servidores_web/{self.node_name}_0 100")


    def scale_up(self):
        if (self.last_active_container_idx + 1) < self.n_max:
            command = f"set weight servidores_web/{self.node_name}_{self.last_active_container_idx+1} 50"
            res = self.send_haproxy_command(command)
            self.last_active_container_idx += 1
            return res


    def scale_down(self):
        if (self.last_active_container_idx) > 0:
            command = f"set weight servidores_web/{self.node_name}_{self.last_active_container_idx} 0"
            res = self.send_haproxy_command(command)
            self.last_active_container_idx -= 1
            return res


    def rebalance_weights(self, weights):
        # Transform normalize weight to 256 base for HAProxy
        weights = [int(w * 256) for w in weights]

        for i in range(len(weights)):
            command = f"set weight servidores_web/{self.node_name}_{i} {weights[i]}"
            self.send_haproxy_command(command)        



    def get_metrics(self) -> list[ContainerMetrics]:
        container_metrics = []
        haproxy_stats_dict = self.get_haproxy_stats()

        for container in self.client.containers.list(filters={"label": "role=lbas_node", "network":"lbas_network"}):
            metric_obj = ContainerMetrics()
            # Gets metrics
            metric = container.stats(stream=False)

            # RAM Metric ---
            ram_usg_bytes = metric["memory_stats"]["usage"]
            ram_limit_bytes = metric["memory_stats"]["limit"]

            ram_usg_pct = ram_usg_bytes / ram_limit_bytes # porcentaje = fraccion_del_total / total
            ram_total_normalize = (ram_limit_bytes / (1024**2)) / self.max_memory # pasamos de b a mb y dividimos por el total (establecido en env)
            
            metric_obj.ram_usg_pct = ram_usg_pct
            metric_obj.ram_total_normalize = ram_total_normalize

            # CPU Metric ---
            cpu_actual_usage = metric["cpu_stats"]["cpu_usage"]["total_usage"]
            cpu_last_usage = metric["precpu_stats"]["cpu_usage"]["total_usage"]
            system_actual_usage = metric["cpu_stats"]["system_cpu_usage"]
            system_last_usage = metric["precpu_stats"]["system_cpu_usage"]
            count_cores = metric["cpu_stats"]["online_cpus"] or 1

            # Getting the cpu_normalize
            delta_cpu = cpu_actual_usage - cpu_last_usage
            delta_system = system_actual_usage - system_last_usage
            
            if delta_system > 0: # Preventing ZeroDivisionError
                cpu_usg = (delta_cpu / delta_system) * count_cores
            else:
                cpu_usg = 0.0

            metric_obj.cpu_usg = cpu_usg

            # Latency & Error Rate Metrics ---
            nombre_actual = container.name # Ej: "lbas_node_0"
            
            if nombre_actual in haproxy_stats_dict:
                metric_obj.latency = haproxy_stats_dict[nombre_actual]["latency"]
                metric_obj.error_rate = haproxy_stats_dict[nombre_actual]["error_rate"]
            else:
                # Fallback por si HAProxy no tiene registrado el nodo aún
                metric_obj.latency = 0.0
                metric_obj.error_rate = 0.0

            # Append to list ---
            container_metrics.append(metric_obj)
        
        return container_metrics

    def init_haproxy_cfg(self):
        with open("haproxy.cfg", "r") as f:
            lines = f.readlines()

        pos = lines.index(f"    balance roundrobin\n")
        new_lines = lines[:pos+1]

        for i in range(self.n_max):
            if i == 0:
                new_lines.append(f"server {self.node_name}_{i} {self.node_name}_{i}:8000 weight 100 check \n")
            else:
                new_lines.append(f"server {self.node_name}_{i} {self.node_name}_{i}:8000 weight 0 check \n")

        with open("haproxy.cfg", "w") as f:
            f.writelines(new_lines)

    def send_haproxy_command(self, command:str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("127.0.0.1", 9999))
            command = command + " \n"
            s.sendall(command.encode("utf-8"))
            res = s.recv(8192).decode("utf-8")

            return res
    
    def get_haproxy_stats(self) -> dict:
        csv_haproxy_res = self.send_haproxy_command("show stat")
        if csv_haproxy_res.startswith("# "):
            csv_haproxy_res = csv_haproxy_res[2:]

        haproxy_stats_dict = {}

        # StringIO convierte un string gigante en un "archivo virtual" para que el módulo csv lo pueda leer
        lector_csv = csv.DictReader(io.StringIO(csv_haproxy_res))

        for fila in lector_csv:
            # Solo nos interesan las filas de los nodos, no las del frontend general
            if fila["pxname"] == "servidores_web":
                nombre_nodo = fila["svname"] # Ej: "lbas_node_0"
                
                # HAProxy devuelve un string vacío '' si no hay datos de latencia aún.
                # Nos aseguramos de convertirlo a 0.0
                latencia = float(fila["rtime"]) if fila["rtime"] else 0.0
                errores = float(fila["hrsp_5xx"]) if fila["hrsp_5xx"] else 0.0
                
                # Guardamos todo en un diccionario usando el nombre del nodo como llave
                haproxy_stats_dict[nombre_nodo] = {
                    "latency": latencia,
                    "error_rate": errores
                }
        
        return haproxy_stats_dict