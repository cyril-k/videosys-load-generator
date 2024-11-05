from typing import List
from prometheus_client import Counter, Gauge, Histogram

class Metrics:
    def __init__(self, labelnames: List[str]):
        self.labels = None
        self.total_gen_count = Counter(
            name="videosys_inference:total_batch_attempts",
            documentation="Number of batch generations attempted",
            labelnames=labelnames,
        )
        self.success_count = Counter(
            name="videosys_inference:successful_batches",
            documentation="Number of successful batch generations",
            labelnames=labelnames,
        )
        self.fail_count = Counter(
            name="videosys_inference:failed_batches",
            documentation="Number of failed batch generations",
            labelnames=labelnames,
        )
        self.total_steps = Counter(
            name="videosys_inference:total_inference_steps",
            documentation="Number of total inference steps",
            labelnames=labelnames,
        )
        self.total_outputs = Counter(
            name="videosys_inference:total_outputs",
            documentation="Number of total outputs generated by the model",
            labelnames=labelnames,
        )
        self.tpb = Histogram(
            name="videosys_inference:tpb",
            documentation="Histogram of time per output batch in seconds",
            labelnames=labelnames + ["bs", ],
            buckets=[
                0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 
                25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0,
                80.0, 90.0, 100.0, 110.0, 120.0,
            ],
        )
        self.tps = Histogram(
            name="videosys_inference:tps",
            documentation="Histogram of time per inference step in seconds",
            labelnames=labelnames + ["bs", ],
            buckets=[
                0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 
                25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0
            ],
        )
        self.bench_duration = Gauge(
            name="videosys_inference:bench_duration",
            documentation="Benchmark duration in seconds",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

    def intitialize_metrics(self):
        assert self.labels is not None, "Provide labels to instantiale metrics"

        # intialize prometheus metrics
        self.total_gen_count.labels(
            **self.labels
        )
        self.success_count.labels(
            **self.labels
        )
        self.fail_count.labels(
            **self.labels
        )
        self.total_outputs.labels(
            **self.labels
        )
        self.total_steps.labels(
            **self.labels
        )
        self.bench_duration.labels(
            **self.labels
        )