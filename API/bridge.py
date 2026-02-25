from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AgentAction(BaseModel):
    weights: list[float]
    decision: float

@app.post("/action")
def post_action(action:AgentAction):
    weights = action.weights
    decision = action.decision

    # Send to proxy update weights

    if decision > 0.7:
        # + 1 container
        pass
    if decision <= 0.3:
        # -1 container
        pass
    
    return {"status": "success", "received": action}


class ContainerMetrics(BaseModel):
    cpu_usg: float
    ram_usg_pct: float
    ram_total_normalize: float
    latency: float
    error_rate: float
    status: bool

@app.get("/metrics")
def get_metrics() -> list[ContainerMetrics]:

    # Get Metrics
    
    metrics = [ContainerMetrics(cpu_usg=0.5, ram_usg_pct=0.4, ram_total_normalize=0.61, 
                                latency=0.12, error_rate=0.08, status=True)]

    return  metrics