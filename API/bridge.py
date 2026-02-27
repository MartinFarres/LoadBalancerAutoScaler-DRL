from fastapi import FastAPI
from pydantic import BaseModel
from ClusterOrchestration import ClusterOrchestration
from schemas import AgentAction, ContainerMetrics

app = FastAPI()
clusterOrchestration = ClusterOrchestration()

@app.post("/init")
def initialize_cluster_orchestration(n_max=10, max_memory=1024, node_name="lbas_node"):
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
def get_metrics() -> list[ContainerMetrics]:
    metrics = clusterOrchestration.get_metrics()
    return  metrics

@app.get("/reset")
def reset():
    clusterOrchestration.reset()

@app.get("/cleanup")
def cleanup():
    clusterOrchestration.stop_all()
    return {"status": "cleaned up"}