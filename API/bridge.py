from fastapi import FastAPI
from pydantic import BaseModel
from ClusterOrchestration import ClusterOrchestration

app = FastAPI()
clusterOrchestration = ClusterOrchestration()

class AgentAction(BaseModel):
    weights: list[float]
    decision: float

class ContainerMetrics(BaseModel):
    cpu_usg: float = 0.0
    ram_usg_pct: float = 0.0
    ram_total_normalize: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    status: bool = False

@app.post("/action")
def post_action(action:AgentAction):
    weights = action.weights
    decision = action.decision

    if decision >= 0.7:
        # + 1 container
        clusterOrchestration.scale_up()

    if decision <= 0.3:
        # -1 container
        clusterOrchestration.scale_down()
    clusterOrchestration.rebalance_weights(weights)
    return {"status": "success", "received": action}

@app.get("/metrics")
def get_metrics(max_memory: float = 1024.0) -> list[ContainerMetrics]:
    metrics = clusterOrchestration.get_metrics(max_memory)
    return  metrics

@app.get("/reset")
def reset():
    clusterOrchestration.reset()