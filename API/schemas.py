from pydantic import BaseModel

class AgentAction(BaseModel):
    weights: list[float]
    decision: float

class ContainerMetrics(BaseModel):
    cpu_usg: float = 0.0
    ram_usg_pct: float = 0.0
    ram_total_normalize: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    status: float = 0.0