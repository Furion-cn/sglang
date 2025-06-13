# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Prometheus Metrics Collection."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import multiprocessing

from sglang.srt.utils import get_bool_env_var

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")
import torch
import logging
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class TimeStats:
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    lb_entry_time: float = 0.0
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0

    class RequestType(Enum):
        UNIFIED = "unified"
        PREFILL = "prefill"
        DECODE = "decode"
        INVALID = "invalid"

    def __str__(self) -> str:
        # if unified
        _type = self.get_type()

        if _type == self.RequestType.UNIFIED:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time}"
        elif _type == self.RequestType.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    bootstrap_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
            return f"bootstrap_duration={self.format_duration(bootstrap_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.prefill_bootstrap_queue_entry_time}"
        # if decode
        elif _type == self.RequestType.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )

            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    prealloc_duration >= 0
                    and transfer_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"prealloc_duration={self.format_duration(prealloc_duration)}, transfer_duration={self.format_duration(transfer_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.decode_prealloc_queue_entry_time}"
        else:
            return "Invalid Time Stats"

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def get_type(self) -> RequestType:
        """Determine the type of request based on timestamp values."""
        if (
            self.prefill_bootstrap_queue_entry_time == 0.0
            and self.prefill_transfer_queue_entry_time == 0.0
            and self.decode_prealloc_queue_entry_time == 0.0
            and self.decode_transfer_queue_entry_time == 0.0
        ):
            return self.RequestType.UNIFIED
        elif (
            self.prefill_bootstrap_queue_entry_time > 0.0
            and self.prefill_transfer_queue_entry_time > 0.0
        ):
            return self.RequestType.PREFILL
        elif (
            self.decode_prealloc_queue_entry_time > 0.0
            and self.decode_transfer_queue_entry_time > 0.0
            and self.wait_queue_entry_time > 0.0
        ):
            return self.RequestType.DECODE
        else:
            return self.RequestType.INVALID


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    cache_hit_rate: float = 0.0
    num_grammar_queue_reqs: int = 0
    spec_accept_length: float = 0.0
    avg_request_queue_latency: float = 0.0
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_infight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0
    input_throughput_schedule_time: float = 0.0
    input_throughput_run_time: float = 0.0


class SchedulerMetricsCollector:

    def __init__(self, tp_rank: int, dp_size: int, labels: Dict[str, str]) -> None:
        self.labels = labels
        self.tp_rank = tp_rank
        self.dp_size = dp_size
        self.last_log_time = time.perf_counter()

        if self.tp_rank == 0:
            self.shared_array = multiprocessing.Array('f', dp_size * 15)
            self.data_lock = multiprocessing.Lock()
            self.data_ready = multiprocessing.Event()
            
            self._metrics_process = multiprocessing.Process(
                target=SchedulerMetricsCollector._metrics_updater_process,
                args=(self.shared_array, self.data_lock, self.data_ready, dp_size, labels)
            )
            self._metrics_process.daemon = True
            self._metrics_process.start()

    @staticmethod
    def _metrics_updater_process(shared_array, data_lock, data_ready, dp_size, labels):
        import numpy as np
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [Metrics Process %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        try:
            process = psutil.Process(os.getpid())
            total_physical_cores = psutil.cpu_count(logical=False)
            total_cores = psutil.cpu_count()
            
            if total_cores > total_physical_cores:
                last_physical_core = total_physical_cores - 1
                corresponding_ht_core = last_physical_core + total_physical_cores
                metrics_cores = [last_physical_core, corresponding_ht_core]
            else:
                metrics_cores = [total_physical_cores - 1]
            
            process.cpu_affinity(metrics_cores)
            logger.info(f"Metrics collector process {os.getpid()} running on CPUs: {process.cpu_affinity()}")

            # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
            from prometheus_client import Counter, Gauge

            labelnames_dp = list(labels.keys())
            labelnames_dp.append("dp")

            num_running_reqs = Gauge(
                name="sglang:num_running_reqs",
                documentation="The number of running requests.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            num_used_tokens = Gauge(
                name="sglang:num_used_tokens",
                documentation="The number of used tokens.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            token_usage = Gauge(
                name="sglang:token_usage",
                documentation="The token usage.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            gen_throughput = Gauge(
                name="sglang:gen_throughput",
                documentation="The generation throughput (token/s).",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            num_queue_reqs = Gauge(
                name="sglang:num_queue_reqs",
                documentation="The number of requests in the waiting queue.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            num_grammar_queue_reqs = Gauge(
                name="sglang:num_grammar_queue_reqs",
                documentation="The number of requests in the grammar waiting queue.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            cache_hit_rate = Gauge(
                name="sglang:cache_hit_rate",
                documentation="The prefix cache hit rate.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            spec_accept_length = Gauge(
                name="sglang:spec_accept_length",
                documentation="The average acceptance length of speculative decoding.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            avg_request_queue_latency = Gauge(
                name="sglang:avg_request_queue_latency",
                documentation="The average request queue latency for the last batch of requests in seconds.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            input_throughput_schedule_time = Gauge(
                name="sglang:input_throughput_schedule_time",
                documentation="The input throughput in schedule time.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            input_throughput_run_time = Gauge(
                name="sglang:input_throughput_run_time",
                documentation="The input throughput in run time.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            # Disaggregation queue metrics
            num_prefill_prealloc_queue_reqs = Gauge(
                name="sglang:num_prefill_prealloc_queue_reqs",
                documentation="The number of requests in the prefill prealloc queue.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            num_prefill_infight_queue_reqs = Gauge(
                name="sglang:num_prefill_infight_queue_reqs",
                documentation="The number of requests in the prefill infight queue.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            num_decode_prealloc_queue_reqs = Gauge(
                name="sglang:num_decode_prealloc_queue_reqs",
                documentation="The number of requests in the decode prealloc queue.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            num_decode_transfer_queue_reqs = Gauge(
                name="sglang:num_decode_transfer_queue_reqs",
                documentation="The number of requests in the decode transfer queue.",
                labelnames=labelnames_dp,
                multiprocess_mode="mostrecent",
            )

            labeled_gauges = {}
            for dp in range(dp_size):
                dp_labels = labels.copy()
                dp_labels["dp"] = str(dp)
                
                labeled_gauges[dp] = {
                    'num_running_reqs': num_running_reqs.labels(**dp_labels),
                    'num_used_tokens': num_used_tokens.labels(**dp_labels),
                    'token_usage': token_usage.labels(**dp_labels),
                    'gen_throughput': gen_throughput.labels(**dp_labels),
                    'num_queue_reqs': num_queue_reqs.labels(**dp_labels),
                    'cache_hit_rate': cache_hit_rate.labels(**dp_labels),
                    'num_grammar_queue_reqs': num_grammar_queue_reqs.labels(**dp_labels),
                    'spec_accept_length': spec_accept_length.labels(**dp_labels),
                    'avg_request_queue_latency': avg_request_queue_latency.labels(**dp_labels),
                    'input_throughput_schedule_time': input_throughput_schedule_time.labels(**dp_labels),
                    'input_throughput_run_time': input_throughput_run_time.labels(**dp_labels),
                    'num_prefill_prealloc_queue_reqs': num_prefill_prealloc_queue_reqs.labels(**dp_labels),
                    'num_prefill_infight_queue_reqs': num_prefill_infight_queue_reqs.labels(**dp_labels),
                    'num_decode_prealloc_queue_reqs': num_decode_prealloc_queue_reqs.labels(**dp_labels),
                    'num_decode_transfer_queue_reqs': num_decode_transfer_queue_reqs.labels(**dp_labels),
                }
            

            shared_np = np.frombuffer(shared_array.get_obj(), dtype=np.float32)
            shared_np = shared_np.reshape((dp_size, 15))

            while True:
                try:
                    if data_ready.wait(timeout=15):
                        data_ready.clear()
                        with data_lock:
                            stats = shared_np.copy()

                        if stats is not None and stats.size > 0:
                            for i, stat in enumerate(stats):
                                if i >= dp_size:
                                    break
                                gauges = labeled_gauges[i]
                                gauges['num_running_reqs'].set(stat[0])
                                gauges['num_used_tokens'].set(stat[1])
                                gauges['token_usage'].set(stat[2])
                                gauges['num_queue_reqs'].set(stat[3])
                                gauges['cache_hit_rate'].set(stat[4])
                                gauges['avg_request_queue_latency'].set(stat[5])
                                gauges['gen_throughput'].set(stat[6])
                                gauges['num_grammar_queue_reqs'].set(stat[7])
                                gauges['spec_accept_length'].set(stat[8])
                                gauges['input_throughput_schedule_time'].set(stat[9])
                                gauges['input_throughput_run_time'].set(stat[10])
                                gauges['num_prefill_prealloc_queue_reqs'].set(stat[11])
                                gauges['num_prefill_infight_queue_reqs'].set(stat[12])
                                gauges['num_decode_prealloc_queue_reqs'].set(stat[13])
                                gauges['num_decode_transfer_queue_reqs'].set(stat[14])
                    else:
                        logger.info("No data received from the main process")
                except Exception as e:
                    logger.warning(f"Metrics updater error: {e}")
                    time.sleep(1)
                
        except Exception as e:
                logger.error(f"Error in metrics collector process: {e}")


    def update_stats(self, stats):
        import numpy as np
        try:
            if not isinstance(stats, np.ndarray):
                logger.warning(f"Invalid stats type: {type(stats)}")
                return
                
            if stats.shape != (self.dp_size, 15):
                logger.warning(f"Invalid stats shape: {stats.shape}, expected ({self.dp_size}, 15)")
                return
            
            if stats.dtype != np.float32:
                stats = stats.astype(np.float32)
            
            shared_np = np.frombuffer(self.shared_array.get_obj(), dtype=np.float32)
            shared_np = shared_np.reshape((self.dp_size, 15))
            
            with self.data_lock:
                np.copyto(shared_np, stats)
            
            self.data_ready.set()
        except Exception as e:
            logger.warning(f"Error in update_stats: {e}")

    def _log_gauge_with_dp(self, gauge, data: Union[int, float], labels: Dict[str, str]) -> None:
        gauge.labels(labels).set(data)

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _stats_to_tensor(self, stats: SchedulerStats) -> torch.Tensor:
        data = torch.zeros(15, dtype=torch.float32)
        data[0] = stats.num_running_reqs
        data[1] = stats.num_used_tokens
        data[2] = stats.token_usage
        data[3] = stats.num_queue_reqs
        data[4] = stats.cache_hit_rate
        data[5] = stats.avg_request_queue_latency
        data[6] = stats.gen_throughput
        data[7] = stats.num_grammar_queue_reqs
        data[8] = stats.spec_accept_length
        data[9] = stats.input_throughput_schedule_time
        data[10] = stats.input_throughput_run_time
        data[11] = stats.num_prefill_prealloc_queue_reqs
        data[12] = stats.num_prefill_infight_queue_reqs
        data[13] = stats.num_decode_prealloc_queue_reqs
        data[14] = stats.num_decode_transfer_queue_reqs
        return data

    def log_stats(self, stats: SchedulerStats, dp_size: int, attn_tp_rank: int, attn_tp_size: int, tp_cpu_group) -> None:
        if attn_tp_rank != 0:
            local_info = torch.zeros(15, dtype=torch.float32, device="cpu")
        else:
            local_info = self._stats_to_tensor(stats)
            if local_info.size(0) != 15:
                raise ValueError(f"local_info.size(0) != 15: {local_info.size(0)}")
            
        global_info = torch.empty(
            (dp_size, attn_tp_size, 15),
            dtype=torch.float32,
            device="cpu",
        )
        
        torch.distributed.all_gather_into_tensor(
            global_info.flatten(),
            local_info,
            group=tp_cpu_group,
        )
        
        # Only log the stats for the first TP rank of the first DP rank
        if attn_tp_rank == 0 and self.tp_rank == 0:
            self.update_stats(global_info[:, 0, :].numpy())
            self.last_log_time = time.perf_counter()

    def __del__(self):
        if hasattr(self, '_metrics_process') and self.tp_rank == 0 and self._metrics_process:
            self._metrics_process.terminate()
            self._metrics_process.join(timeout=1)

class TokenizerMetricsCollector:
    def __init__(
        self,
        labels: Dict[str, str],
        bucket_time_to_first_token: Optional[List[float]] = None,
        bucket_inter_token_latency: Optional[List[float]] = None,
        bucket_e2e_request_latency: Optional[List[float]] = None,
        collect_tokens_histogram: bool = False,
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels
        self.collect_tokens_histogram = collect_tokens_histogram

        self.prompt_tokens_total = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )

        self.generation_tokens_total = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )

        if collect_tokens_histogram:
            bucket_prompt_tokens = [
                100,
                300,
                500,
                700,
                1000,
                1500,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
                12000,
                15000,
                20000,
                22000,
                25000,
                30000,
                35000,
                40000,
            ]
            self.prompt_tokens_histogram = Histogram(
                name="sglang:prompt_tokens_histogram",
                documentation="Histogram of prompt token length.",
                labelnames=labels.keys(),
                buckets=bucket_prompt_tokens,
            )
            bucket_generation_tokens = [
                100,
                300,
                500,
                1000,
                1200,
                1500,
                1700,
                2000,
                2500,
                3000,
                3500,
                4000,
                4500,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
            ]
            self.generation_tokens_histogram = Histogram(
                name="sglang:generation_tokens_histogram",
                documentation="Histogram of generation token length.",
                labelnames=labels.keys(),
                buckets=bucket_generation_tokens,
            )

        self.cached_tokens_total = Counter(
            name="sglang:cached_tokens_total",
            documentation="Number of cached prompt tokens.",
            labelnames=labels.keys(),
        )

        self.num_requests_total = Counter(
            name="sglang:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.num_so_requests_total = Counter(
            name="sglang:num_so_requests_total",
            documentation="Number of structured output requests processed.",
            labelnames=labels.keys(),
        )

        if bucket_time_to_first_token is None:
            bucket_time_to_first_token = [
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
            ]

        if bucket_e2e_request_latency is None:
            bucket_e2e_request_latency = [
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
                800,
            ]

        if bucket_inter_token_latency is None:
            bucket_inter_token_latency = [
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
                4.000,
                6.000,
                8.000,
            ]

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_time_to_first_token,
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_inter_token_latency,
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=bucket_e2e_request_latency,
        )

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
        has_grammar: bool,
    ):
        self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
        if cached_tokens > 0:
            self.cached_tokens_total.labels(**self.labels).inc(cached_tokens)
        self.num_requests_total.labels(**self.labels).inc(1)
        if has_grammar:
            self.num_so_requests_total.labels(**self.labels).inc(1)
        self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)
        if self.collect_tokens_histogram:
            self._log_histogram(self.prompt_tokens_histogram, prompt_tokens)
            self._log_histogram(self.generation_tokens_histogram, generation_tokens)

    def observe_time_to_first_token(self, value: float):
        self.histogram_time_to_first_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(self, interval: float, num_new_tokens: int):
        adjusted_interval = interval / num_new_tokens

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
        his._sum.inc(interval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break
