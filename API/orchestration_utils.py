import docker
from bridge import ContainerMetrics

client = docker.from_env()
image = client.images.get("My Image")
# Pull image or Create image in dockerfile with a functioning server

def start(n_max):
    # Stop all containers
    stop_all()
    for _ in range(n_max):
        client.containers.run(image=image, detach=True, labels={"role": "lbas_node"})

    # TODO: Set all conteiner weights in HAProxy to 0 except for one
    
def stop_all():
    for container in client.containers.list(filters={"label": "role=lbas_node"}):
        container.stop()

def scale_up():
    # client.containers.run(image=image, detach=True, labels={"role": "lbas_node"})

    # TODO: Update a container's weight from 0 to 50

    pass

def scale_down():
    # containers = client.containers.list(filters={"label": "role=lbas_node"})
    # if len(containers) > 1:
    #     containers[-1].stop()

    # TODO: Update a container's weight from X to 0
    pass

def get_metrics(max_memory) -> list[ContainerMetrics]:
    container_metrics = []
    for container in client.containers.list(filters={"label": "role=lbas_node"}):
        metric_obj = ContainerMetrics()
        # Gets metrics
        metric = container.stats(stream=False)

        # RAM Metric ---
        ram_usg_bytes = metric["memory_stats"]["usage"]
        ram_limit_bytes = metric["memory_stats"]["limit"]

        ram_usg_pct = ram_usg_bytes / ram_limit_bytes # porcentaje = fraccion_del_total / total
        ram_total_normalize = (ram_limit_bytes / (1024**2)) / max_memory # pasamos de b a mb y dividimos por el total (establecido en env)
        
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

        # Append to list ---
        container_metrics.append(metric_obj)
    
    return container_metrics
