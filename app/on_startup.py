import os
import time
import copy
import logging 
import requests
import aiohttp

from metrics import Metrics

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8000"
RETRY_INTERVAL = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

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

    check_videosys_health()

    api_url = f"{SERVER_URL}/generate"
    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:


        for batch_size in range(1, 9):
            for num_steps in range (20, 110, 10):
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
                        print(await response.json())
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
                finally:
                    metrics.total_gen_count.labels(**metrics.labels).inc()
                    metrics.total_steps.labels(**metrics.labels).inc(num_steps)
                    metrics.total_outputs.labels(**metrics.labels).inc(batch_size)
    end_time = time.perf_counter()
    metrics.bench_duration.labels(**metrics.labels).set(end_time - start_time)