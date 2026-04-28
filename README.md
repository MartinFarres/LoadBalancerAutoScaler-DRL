# Load Balancer Auto-Scaler with Deep Reinforcement Learning (PPO)

An autonomous cluster orchestration system that uses **Proximal Policy Optimization (PPO)** to simultaneously manage **load balancing** and **horizontal auto-scaling** for a Docker-based microservice cluster. The agent learns to dynamically distribute traffic weights and scale containers up or down in response to real-time workload metrics.

---

## Architecture Overview

<!-- ```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE                            │
│   PPO Agent ──POST /action──► Bridge ──GET /metrics──► PPO Agent│
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │     Bridge (FastAPI)        │
          │  ClusterOrchestration.py    │
          └──────┬─────────────┬───────┘
                 │             │
    ┌────────────▼──┐    ┌─────▼──────────┐
    │  Docker Daemon│    │    HAProxy      │
    │  (cgroups:    │    │  (L7 Routing,  │
    │  CPU / RAM)   │    │  Latency,Errors)│
    └────────────┬──┘    └─────┬──────────┘
                 │             │
          ┌──────▼─────────────▼──────┐
          │   Docker Container Cluster │
          │  lbas_node_0 … lbas_node_N │
          └───────────────────────────┘
                        ▲
               HTTP Traffic (Locust)
``` -->

<img width="1287" height="622" alt="image" src="https://github.com/user-attachments/assets/57d54f1e-42ff-45e5-864f-efd5ca485df1" />

The system is composed of four layers:

1. **Control Plane** — The PPO agent sends an action vector (routing weights + scale decision) to the Bridge API, and receives back the current observation space (per-container metrics).
2. **Infrastructure Management** — The Bridge uses the Docker SDK to start/stop containers and reads hardware metrics via Linux cgroups.
3. **L7 Routing & Monitoring** — HAProxy distributes traffic according to the weights set by the agent and reports latency and HTTP error rates.
4. **Data Plane** — Locust generates realistic, variable HTTP traffic patterns against the cluster entry point.

---

## Training Pipeline

Training is split into two phases to maximize sample efficiency:

| Phase       | Mode               | Description                                                                                         |
| ----------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| **Phase 1** | Simulated          | Pre-trains the agent using a mathematical M/M/1 queueing model. Fast iteration, no Docker required. |
| **Phase 2** | Real (Fine-Tuning) | Fine-tunes the pre-trained model on a live Docker cluster with real Locust traffic.                 |

The simulated environment models CPU, RAM (Little's Law), latency (M/M/1), and error rate (Sigmoid overflow) with Gaussian noise and stochastic traffic patterns (double wave, linear, exponential, step functions).

---

## Project Structure

```
.
├── main.py                          # Main entrypoint — run all pipelines from here
├── requirements.txt
├── cleanup.sh                       # Kill orphan processes and Docker containers
│
├── API/
│   ├── bridge.py                    # FastAPI bridge between agent and Docker cluster
│   ├── ClusterOrchestration.py      # Docker SDK + HAProxy + cgroups metrics collector
│   ├── schemas.py                   # Pydantic models (AgentAction, ContainerMetrics)
│   ├── fast_metrics.c               # C shared library for fast network namespace metric reads
│   ├── haproxy.cfg                  # Auto-generated HAProxy configuration
│   ├── locust.py                    # Locust load generator with dynamic traffic shapes
│   └── dummy_server/
│       ├── app.py                   # Flask server with /, /cpu, /ram endpoints
│       ├── dockerfile
│       └── requirements.txt
│
└── environment/
    ├── environment.py               # Gymnasium environment (real + simulated modes)
    ├── train_agent.py               # Phase 1 & Phase 2 training scripts
    ├── test_agent.py                # Run inference with a trained PPO model
    ├── baseline_agent.py            # Industry baseline (Round Robin + CPU thresholds)
    ├── baseline_PID.py              # Classical control baseline (PID controller)
    ├── callbacks.py                 # Custom SB3 callback — logs metrics to CSV + TensorBoard
    ├── visualizer.py                # Plots learning curves and testing summary tables
    └── test_env.py                  # Gymnasium environment sanity checker
```

---

## Prerequisites

- Python 3.10+
- Docker Engine running (the Bridge uses the Docker socket)
- Build the dummy server image before running in real mode:

```bash
cd API/dummy_server
docker build -t dummy_server:latest .
```

- Compile the C metrics library:

```bash
cd API
gcc -shared -fPIC -o libfastmetrics.so fast_metrics.c
```

- Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

All pipelines are launched through `main.py`.

```bash
python main.py --pipeline <PIPELINE> [OPTIONS]
```

### Run the full pipeline (recommended first run)

```bash
python main.py --pipeline all
```

This executes in sequence: simulated pre-training → real fine-tuning → baseline comparison tests.

---

## `main.py` — Arguments Reference

| Argument                 | Short | Type  | Default                | Description                                                  |
| ------------------------ | ----- | ----- | ---------------------- | ------------------------------------------------------------ |
| `--pipeline`             | `-p`  | `str` | `all`                  | Which pipeline to run. See options below.                    |
| `--nodes`                | `-n`  | `int` | `5`                    | Number of Docker containers in the cluster. Must be ≤ 10.    |
| `--file`                 | `-f`  | `str` | `training_metrics.csv` | Output CSV filename for metrics (a prefix is auto-appended). |
| `--simulated_iterations` | `-si` | `int` | `50000`                | Timesteps for Phase 1 simulated training.                    |
| `--real_iterations`      | `-ri` | `int` | `5000`                 | Timesteps for Phase 2 real-world fine-tuning.                |
| `--testing_iterations`   | `-ti` | `int` | `1000`                 | Steps for the PPO agent evaluation run.                      |
| `--pid_iterations`       | `-pi` | `int` | `1000`                 | Steps for the PID baseline evaluation.                       |
| `--agent_iterations`     | `-ai` | `int` | `1000`                 | Steps for the industry baseline evaluation.                  |

### `--pipeline` options

| Value           | Description                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------- |
| `simulado`      | Phase 1 only — trains the agent in the fast mathematical simulator. No Docker required.       |
| `real`          | Phase 2 only — fine-tunes an existing model on a live Docker cluster. Requires Phase 1 model. |
| `test_ppo`      | Runs inference with the trained PPO model and generates a metrics summary table.              |
| `test_baseline` | Runs the industry baseline (Round Robin + static CPU thresholds).                             |
| `test_pid`      | Runs the PID controller baseline.                                                             |
| `all`           | Runs the entire pipeline: Phase 1 → Phase 2 → all three test comparisons.                     |

### Examples

```bash
# Fast simulated pre-training only, 10 nodes, 100k steps
python main.py -p simulado -n 10 -si 100000

# Real fine-tuning only, 5 nodes, 2000 steps
python main.py -p real -n 5 -ri 2000

# Evaluate the trained PPO agent for 3000 steps
python main.py -p test_ppo -n 5 -ti 3000

# Run only the industry baseline comparison
python main.py -p test_baseline -n 5 -ai 5000

# Full pipeline with custom node count and iterations
python main.py -p all -n 8 -si 80000 -ri 4000 -ti 2000
```

---

## Running Scripts Directly

Each script in `environment/` can also be run standalone.

### Train — Phase 1 (Simulated)

```bash
python environment/train_agent.py train_phase_1_simulation \
  --nodes 5 \
  --iterations 50000 \
  --file training_metrics.csv
```

### Train — Phase 2 (Real Docker cluster)

> Requires the Bridge API and Docker cluster to be running first.

```bash
python environment/train_agent.py train_phase_2_real_world \
  --nodes 5 \
  --iterations 5000 \
  --file training_metrics.csv
```

### Test the PPO Agent

```bash
python environment/test_agent.py \
  --nodes 5 \
  --iterations 5000 \
  --file testing_metrics.csv
```

### Test Baselines

```bash
# Industry baseline (Round Robin + thresholds)
python environment/baseline_agent.py --nodes 5 --iterations 5000

# PID controller baseline
python environment/baseline_PID.py --nodes 5 --iterations 5000
```

### Verify the Gymnasium Environment

```bash
python environment/test_env.py
```

---

## Bridge API Endpoints

The Bridge (`API/bridge.py`) exposes a FastAPI server on port `8000` that acts as the middleware between the PPO agent and the Docker cluster.

| Method | Endpoint   | Description                                                            |
| ------ | ---------- | ---------------------------------------------------------------------- |
| `POST` | `/init`    | Initializes the cluster. Starts all containers and HAProxy.            |
| `POST` | `/action`  | Receives the agent's action (weights + scale decision) and applies it. |
| `GET`  | `/metrics` | Returns the current `ContainerMetrics` for all N containers.           |
| `GET`  | `/reset`   | Resets the cluster to its initial state (1 active container).          |
| `GET`  | `/cleanup` | Stops and removes all managed Docker containers.                       |

#### `/init` parameters (query string)

| Parameter    | Default     | Description                                |
| ------------ | ----------- | ------------------------------------------ |
| `n_max`      | `10`        | Maximum number of containers to provision. |
| `max_memory` | `1024`      | Memory limit per container in MB.          |
| `node_name`  | `lbas_node` | Prefix name for container instances.       |

#### `/action` body (`AgentAction`)

```json
{
  "weights": [0.5, 0.3, 0.2, 0.0, ...],
  "decision": 0.8
}
```

- `weights`: List of N floats `[0.0, 1.0]`. Routing weights for each container slot. Values are scaled to HAProxy's 0–256 weight range internally.
- `decision`: Float `[0.0, 1.0]`. Scale-up if `≥ 0.7`, scale-down if `≤ 0.3`, hold otherwise.

---

## Observation Space & Action Space

### Observation Space

A flat vector of shape `(n_max × 6,)` where each group of 6 values represents one container slot:

| Index (offset) | Metric                | Range    | Description                           |
| -------------- | --------------------- | -------- | ------------------------------------- |
| `+0`           | `cpu_usg`             | `[0, 1]` | CPU utilization                       |
| `+1`           | `ram_usg_pct`         | `[0, 1]` | RAM usage as % of limit               |
| `+2`           | `ram_total_normalize` | `[0, 1]` | Normalized RAM limit                  |
| `+3`           | `latency`             | `[0, 1]` | Response time (normalized to 2000 ms) |
| `+4`           | `error_rate`          | `[0, 1]` | HTTP 5xx error rate                   |
| `+5`           | `status`              | `{0, 1}` | Container active (1) or off (0)       |

### Action Space

A flat vector of shape `(n_max + 1,)`:

| Indices         | Content                       | Range    |
| --------------- | ----------------------------- | -------- |
| `[0 … n_max-1]` | Routing weights per container | `[0, 1]` |
| `[n_max]`       | Scale decision                | `[0, 1]` |

---

## Reward Function

The reward is a penalty-based signal designed to balance service quality against infrastructure cost:

```
R = -(W_latency × avg_latency² + W_errors × avg_errors + W_cost × (active/n_max) + W_saturation)
```

| Component      | Weight | Penalizes                                      |
| -------------- | ------ | ---------------------------------------------- |
| Latency        | 2.0    | High response times (squared to punish spikes) |
| Errors         | 50.0   | HTTP 5xx responses                             |
| Cost           | 1.0    | Unnecessary active containers                  |
| CPU Saturation | 1.0    | CPU usage above 80%                            |
| RAM Saturation | 1.0    | RAM usage above 85%                            |
| Scaling churn  | 0.05   | Frequent unnecessary scale events              |
| Total failure  | −200   | All containers offline                         |

---

## Monitoring

### TensorBoard

```bash
tensorboard --logdir ./logs_tensorboard/
# Open http://localhost:6006
```

Training curves, policy loss, and value loss are logged automatically during both training phases.

### Output Files

| Path                                  | Contents                                                  |
| ------------------------------------- | --------------------------------------------------------- |
| `./training_results/phase1/`          | Phase 1 training metrics CSV                              |
| `./training_results/phase2/`          | Phase 2 fine-tuning metrics CSV                           |
| `./training_results/testing_results/` | Per-run test metrics CSVs                                 |
| `./resultados_graficos/`              | Learning curve PNGs and summary table images              |
| `./logs_checkpoints/`                 | SB3 model checkpoints (saved every 2000 steps in Phase 2) |
| `ppo_lb_simulated_base.zip`           | Saved Phase 1 model                                       |
| `ppo_lb_production_ready.zip`         | Saved Phase 2 model                                       |

---

## Cleanup

If processes or containers are left behind after a crash or manual interruption:

```bash
bash cleanup.sh
```

This will kill any orphaned `uvicorn`, `locust`, `tensorboard`, and training processes, free ports `8000`, `6006`, and `8089`, and remove all `lbas_*` Docker containers and the `lbas_network`.

---

## Baselines for Comparison

Two baselines are implemented for benchmarking against the PPO agent:

**Industry Baseline** (`baseline_agent.py`) — Replicates the most common production approach:

- Load balancing: Round Robin (equal weights for all active nodes)
- Auto-scaling: Static CPU thresholds (scale up if CPU > 75%, scale down if CPU < 25%)

**PID Controller Baseline** (`baseline_PID.py`) — Classical control theory approach:

- Maintains average CPU usage at a 60% setpoint using a PID controller (Kp=1.5, Ki=0.1, Kd=0.5)
- Load balancing: Round Robin

---

## Bibliography

1. Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). The MIT Press.
2. Russell, S., & Norvig, P. (2010). _Artificial Intelligence: A Modern Approach_ (3rd ed.). Prentice Hall.
3. Harchol-Balter, M. (2013). _Performance Modeling and Design of Computer Systems: Queueing Theory in Action_. Cambridge University Press.
4. Tesauro, G. et al. (2006). A hybrid reinforcement learning approach to autonomic resource allocation. _IEEE ICAC 2006_.
5. Menascé, D. A., & Almeida, V. A. F. (2001). _Capacity Planning for Web Services_. Prentice Hall.
