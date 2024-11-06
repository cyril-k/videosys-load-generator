import os
import time
import copy
import logging 
import requests
import aiohttp
from datetime import datetime

from metrics import Metrics
from utils import (
    save_result,
    get_instance_name,
    get_specs
)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8000"
RETRY_INTERVAL = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
SPEC = {
    "served_model": os.environ.get("SERVED_MODEL", "THUDM/CogVideoX-5b"),
    "batch_size_list": list(range(1, 9)),
    "num_inference_steps_list": list(range(20, 110, 10)),
}

def check_videosys_health():
    while True:
        try:
            response = requests.get(f"{SERVER_URL}/initialize", timeout=5)
            if response.status_code == 200:
                return
            else:
                print(f"Server returned status code {response.status_code}. Retrying...")
                
        except requests.exceptions.RequestException as e:
            print(f"Could not connect to the remote server: {e}. Retrying in {RETRY_INTERVAL} seconds...")
        
        time.sleep(RETRY_INTERVAL)


async def run_benchmark(metrics: Metrics):
    spec = get_specs() or SPEC
    served_model = spec.get("served_model")
    instance_name = get_instance_name()
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    result_file_name = f"videosys_{metrics.labels['model']}_{instance_name}_{timestamp}"
    results = []

    check_videosys_health()

    api_url = f"{SERVER_URL}/generate"
    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        for batch_size in spec.get("batch_size_list"):
            for num_steps in spec.get("num_inference_steps_list"):
                payload = {
                    "prompt": [
                        "Sunset over the sea.",
                        ] * batch_size,
                    "num_inference_steps": num_steps,
                    "seed": -1,
                    "cfg": 6.5,
                    "save_disk_path": "./results"
                }
                headers = {
                    "Content-Type": "application/json",
                }
                logger.info(
                    f"Running inferencem with batch of {len(payload['prompt'])} prompts and num_inference_steps of {num_steps}"
                )
                try:
                    async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                        response.raise_for_status()
                        inference_time = float((await response.json())["elapsed_time"])
                        histogram_labels = copy.deepcopy(metrics.labels)
                        histogram_labels["bs"] = batch_size
                        metrics.tpb.labels(**histogram_labels).observe(
                            inference_time
                        )
                        metrics.tps.labels(**histogram_labels).observe(
                            inference_time / num_steps
                        )
                        metrics.success_count.labels(**metrics.labels).inc()
                except Exception as e:
                    logger.exception(e)
                    metrics.fail_count.labels(**metrics.labels).inc()
                    inference_time = None
                finally:
                    metrics.total_gen_count.labels(**metrics.labels).inc()
                    metrics.total_steps.labels(**metrics.labels).inc(num_steps)
                    metrics.total_outputs.labels(**metrics.labels).inc(batch_size)
                    results.append(
                        {   
                            "served_model": served_model,
                            "batch_size": batch_size,
                            "num_inference_steps": num_steps,
                            "inference_time": inference_time,
                            "payload": payload
                        }
                    )
    end_time = time.perf_counter()
    benchmark_duration = end_time - start_time
    metrics.bench_duration.labels(**metrics.labels).set(benchmark_duration)
    save_result(
        result_dict={
            "instance_name": instance_name,
            "bencmark_duration": benchmark_duration,
            "results": results
        },
        file_name=result_file_name,
    )