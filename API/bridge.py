from fastapi import FastAPI
from pydantic import BaseModel
from ClusterOrchestration import ClusterOrchestration
from schemas import AgentAction, ContainerMetrics, ClusterMetrics

app = FastAPI()
clusterOrchestration = ClusterOrchestration()

@app.post("/init")
def initialize_cluster_orchestration(n_max: int = 10, max_memory: int = 1024, node_name: str = "lbas_node"):
    clusterOrchestration.set_params_and_start(n_max, max_memory, node_name)
    return {"status": "initialized", "n_max": n_max}

@app.post("/action")
def post_action(action:AgentAction):
    weights = action.weights # LB -> [w1, w2, w3, ..., wn]
    decision = action.decision # AS -> 1.0 | 0.0

    if decision >= 0.7:
        # + 1 container
        clusterOrchestration.scale_up()

    if decision <= 0.3:
        # -1 container
        clusterOrchestration.scale_down()
        
    clusterOrchestration.rebalance_weights(weights)
    return {"status": "success", "received": action}

@app.get("/metrics")
def get_metrics() -> ClusterMetrics:
    nodes = clusterOrchestration.get_metrics()
    # workload_norm = clusterOrchestration.get_workload_norm()
    # Usa el user_count que Locust reporta, no el req_rate de HAProxy
    workload_norm = getattr(clusterOrchestration, 'last_workload_norm', 0.0)
    return ClusterMetrics(nodes=nodes, workload_norm=workload_norm)

@app.post("/workload")
def post_workload(user_count: int, total_users: int = 4000):
    """Called by Locust tick() to report current user count."""
    clusterOrchestration.last_workload_norm = min(1.0, user_count / total_users)
    return {"status": "ok"}

@app.get("/reset")
def reset():
    clusterOrchestration.reset()

@app.get("/cleanup")
def cleanup():
    clusterOrchestration.stop_all()
    return {"status": "cleaned up"}