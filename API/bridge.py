from fastapi import FastAPI
from pydantic import BaseModel
import API.orchestration_utils as orchestration_utils

app = FastAPI()

class AgentAction(BaseModel):
    weights: list[float]
    decision: float

@app.post("/action")
def post_action(action:AgentAction):
    weights = action.weights
    decision = action.decision

    if decision > 0.7:
        # + 1 container
        orchestration_utils.scale_up()

    if decision <= 0.3:
        # -1 container
        orchestration_utils.scale_down()


    # TODO: Send to HAProxy update weights command

    
    return {"status": "success", "received": action}


class ContainerMetrics(BaseModel):
    cpu_usg: float = 0.0
    ram_usg_pct: float = 0.0
    ram_total_normalize: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    status: bool = False

@app.get("/metrics")
def get_metrics(max_memory) -> list[ContainerMetrics]:

    metrics_docker = orchestration_utils.get_metrics(max_memory) 

    # TODO: Get Latency & Error rate from HAProxy

    return  metrics