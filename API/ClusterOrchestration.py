import docker
import socket
import csv
import io
import os
import concurrent.futures
import ctypes
import time
from schemas import ContainerMetrics


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
        self.image_container = self._get_or_pull("dummy_server:latest")
        
        # HAProxy image
        self.image_HAProxy = self._get_or_pull("haproxytech/haproxy-alpine:3.0")
        
        # Fast Metrics C Collector
        self.lib = ctypes.CDLL('./libfastmetrics.so')
        self.lib.get_container_rx_bytes.argtypes = [ctypes.c_int, ctypes.c_char_p] # Args expected
        self.lib.get_container_rx_bytes.restype = ctypes.c_longlong # Return type expected
        self.containersPIDs = []
        self.containersLongIDs = []

        # CPU dic for cpu_usg calculation
        self.last_cpu_stats = {} # (cpu_ns, timestamp_ns)

        self.start()

    def _get_or_pull(self, image_name: str):
        """Try to get a local image; if not found, pull it from Docker Hub."""
        try:
            return self.client.images.get(image_name)
        except docker.errors.ImageNotFound:
            print(f"  Image '{image_name}' not found locally. Pulling...")
            return self.client.images.pull(image_name)

    def start(self):
        # Always clean up any leftover containers from previous runs
        self.stop_all()

        # Create network if it doesn't exist
        try:
            self.client.networks.get("lbas_network")
        except docker.errors.NotFound:
            self.client.networks.create("lbas_network")

        # Creates all n containers
        for i in range(self.n_max):
            self.client.containers.run(image=self.image_container, 
                                       network="lbas_network", 
                                       detach=True, 
                                       name=f"{self.node_name}_{i}",
                                       nano_cpus=500000000, # limits container to 50% of the cpus capacity
                                       labels={"role": "lbas_node"})
        
           
        # Creates HAProxy container and its configuration
        self.init_haproxy_cfg()
        self.client.containers.run(image=self.image_HAProxy, 
                                                network="lbas_network", 
                                                name=f"lbas_haproxy", 
                                                volumes={
                                                    # path on your machine/host (must be absolute for Docker SDK)
                                                    os.path.abspath("haproxy.cfg"): {
                                                        "bind": "/usr/local/etc/haproxy/haproxy.cfg",  # path inside the container
                                                        "mode": "rw",
                                                    },
                                                },
                                                ports={'80/tcp': 80, '9999/tcp': 9999},
                                                detach=True,
                                                labels={"role":"lbas_haproxy"})
        
        # Gets all containers PIDs
        for i in range(self.n_max):
            container_attrs = self.client.containers.get(f"{self.node_name}_{i}").attrs
            long_id = container_attrs['Id']
            self.containersPIDs.append(container_attrs['State']['Pid'])
            self.containersLongIDs.append(container_attrs['Id']) # Cache the 64-char Long ID

            # Inicializamos el trackeo de CPU para este contenedor
            self.last_cpu_stats[long_id] = {
                "cpu_usage_ns": 0,
                "timestamp_ns": time.time_ns()
            }

        
        
        
    def stop_all(self):
        # Stop and remove all node containers (including stopped ones)
        for container in self.client.containers.list(all=True, filters={"label": "role=lbas_node"}):
            try:
                container.stop()
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass
        # Stop and remove HAProxy
        try:
            haproxy = self.client.containers.get("lbas_haproxy")
            haproxy.stop()
            haproxy.remove(force=True)
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
        
        for i in range(self.n_max):

            if i <= self.last_active_container_idx:
                final_weight = weights[i]
            else:
                final_weight = 0.0

            command = f"set weight servidores_web/{self.node_name}_{i} {final_weight}"
            self.send_haproxy_command(command)        



    def get_metrics(self) -> list[ContainerMetrics]:
        # Pre-armamos una lista vacía con 10 espacios
        container_metrics = [ContainerMetrics()] * self.n_max
        haproxy_stats_dict = self.get_haproxy_stats()

        # Lanzamos un ThreadPool con la misma cantidad de workers que contenedores
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_max) as executor:
            # Programamos las 10 tareas en paralelo
            futures = [
                executor.submit(self._fetch_single_container_metrics, i, haproxy_stats_dict) 
                for i in range(self.n_max)
            ]

            # A medida que los hilos van terminando, guardamos el resultado en su posición exacta
            for future in concurrent.futures.as_completed(futures):
                idx, metric_obj = future.result()
                container_metrics[idx] = metric_obj
        
        return container_metrics
    

    def _fetch_single_container_metrics(self, i: int, haproxy_stats_dict: dict):
        """Método worker que se ejecutará en paralelo para cada contenedor"""
        metric_obj = ContainerMetrics()
        nombre_nodo = f"{self.node_name}_{i}"
        
        try:
            target_pid = self.containersPIDs[i]
            long_id = self.containersLongIDs[i]
            
            # Metricas de red via C
            interface_name = b"eth0" 
            rx_bytes = self.lib.get_container_rx_bytes(target_pid, interface_name)
            
            if rx_bytes != -1:
                metric_obj.network_rx = rx_bytes
                pass

            
            # Metricas de memoria usando File I/O Cgroups
            # Si es una version mas vieja de linux puede ser: /sys/fs/cgroup/memory/docker/{long_id}/memory.usage_in_bytes
            mem_path = f"/sys/fs/cgroup/system.slice/docker-{long_id}.scope/memory.current"
            
            if os.path.exists(mem_path):
                with open(mem_path, 'r') as f:
                    raw_mem = f.read().strip()
                    # Verificamos que no esté vacío antes de convertir
                    if raw_mem.isdigit():
                        ram_usg_bytes = int(raw_mem)
                        ram_limit_bytes = self.max_memory * 1024 * 1024
                        
                        metric_obj.ram_usg_pct = ram_usg_bytes / ram_limit_bytes 
                        metric_obj.ram_total_normalize = (ram_limit_bytes / (1024**2)) / self.max_memory
           
            # Metricas CPU via Cgroups File I/O
            # Si usa v1: /sys/fs/cgroup/cpuacct/docker/{long_id}/cpuacct.stat
            cpu_path = f"/sys/fs/cgroup/system.slice/docker-{long_id}.scope/cpu.stat"
            if os.path.exists(cpu_path):
                # Leemos el archivo  
                with open(cpu_path, 'r') as f:
                    lines = f.readlines()
                
                usage_usec = 0
                for line in lines:
                    if line.startswith("usage_usec"): # buscamos 'usage_usec'
                        parts = line.split()
                        # parts será una lista, ej: ['usage_usec', '123456']
                        # Aseguramos que tenga al menos 2 elementos para evitar OutOfBounds
                        if len(parts) >= 2 and parts[1].isdigit():
                            usage_usec = int(parts[1]) 
                        break
                
                # Convertimos a nanosegundos para tener la máxima precisión
                current_cpu_ns = usage_usec * 1000
                current_time_ns = time.time_ns()
                
                # Recuperamos los valores del step/iteración anterior
                last_stats = self.last_cpu_stats.get(long_id, {"cpu_usage_ns": 0, "timestamp_ns": current_time_ns})
                last_cpu_ns = last_stats["cpu_usage_ns"]
                last_time_ns = last_stats["timestamp_ns"]

                # Calculamos los deltas
                delta_cpu = current_cpu_ns - last_cpu_ns
                delta_time = current_time_ns - last_time_ns
                
                # Calculamos el uso
                # (delta_cpu / delta_tiempo_real) equivale a
                # (delta_cpu / delta_system) * count_cores. Nos da el uso por core.
                # 1.0 significa 100% de 1 core.
                if delta_time > 0 and last_cpu_ns > 0:
                    metric_obj.cpu_usg = delta_cpu / delta_time
                else:
                    metric_obj.cpu_usg = 0.0
                
                self.last_cpu_stats[long_id] = {
                    "cpu_usage_ns": current_cpu_ns,
                    "timestamp_ns": current_time_ns
                }

        except Exception as e:
            print(f"Error fetching metrics for node {i}: {e}")
            pass

        # L7 METRICS de HAProxy
        if nombre_nodo in haproxy_stats_dict:
            metric_obj.latency = haproxy_stats_dict[nombre_nodo]["latency"]
            metric_obj.error_rate = haproxy_stats_dict[nombre_nodo]["error_rate"]
            metric_obj.status = haproxy_stats_dict[nombre_nodo]["status"]
        else:
            metric_obj.latency = 0.0
            metric_obj.error_rate = 0.0
            metric_obj.status = 0.0

        return i, metric_obj

    def init_haproxy_cfg(self):
        # Base cofiguration hardcodeada
        new_lines = [
            "global\n",
            "    stats socket ipv4@0.0.0.0:9999 level admin\n",
            "    maxconn 2000\n",
            "defaults\n",
            "    mode http\n",
            "    timeout connect 5000ms\n",
            "    timeout client  50000ms\n",
            "    timeout server  50000ms\n",
            "frontend http_in\n",
            "    bind *:80\n",
            "    default_backend servidores_web\n",
            "backend servidores_web\n",
            "    balance roundrobin\n"
        ]

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
                latencia = float(fila["rtime"]) if fila.get("rtime") else 0.0
                errores = float(fila["hrsp_5xx"]) if fila.get("hrsp_5xx") else 0.0
                try:
                    status = 1.0 if int(fila.get("weight") or 0) > 0 else 0.0
                except (ValueError, TypeError):
                    status = 0.0
                
                # Guardamos todo en un diccionario usando el nombre del nodo como llave
                haproxy_stats_dict[nombre_nodo] = {
                    "latency": latencia,
                    "error_rate": errores,
                    "status": status
                }
        
        return haproxy_stats_dict