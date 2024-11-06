import os
import logging
import re
import asyncio

from fastapi import FastAPI
from prometheus_client import make_asgi_app


from metrics import Metrics
from on_startup import run_benchmark
from utils import sanitize_label

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Prometheus metrics
metrics = Metrics(
    labelnames=[
        "model",
    ]
)

app = FastAPI()


async def benchmark_task():
    global metrics
    served_model = os.environ.get("SERVED_MODEL", "THUDM/CogVideoX-5b")

    metrics.labels = {
        "model": sanitize_label(served_model),
    }
    metrics.intitialize_metrics()
    
    await run_benchmark(metrics)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(benchmark_task())

app.mount("/metrics", make_asgi_app())
