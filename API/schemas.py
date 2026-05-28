from pydantic import BaseModel

class AgentAction(BaseModel):
    weights: list[float]
    decision: float

class ContainerMetrics(BaseModel):
    cpu_usg: float = 0.0
    ram_usg_pct: float = 0.0
    queue_depth: float = 0.0   # raw HAProxy qcur (requests queued for this server); normalized in the env
    latency: float = 0.0
    error_rate: float = 0.0
    status: float = 0.0

class ClusterMetrics(BaseModel):
    nodes: list[ContainerMetrics]
    workload_norm: float = 0.0