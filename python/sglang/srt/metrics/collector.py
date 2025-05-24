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

import json
import time
from dataclasses import dataclass
from typing import Dict, Union, Optional
import torch

@dataclass
class EPLBManagerStats:
    rebalance_total_time: float
    rebalance_compute_time: float
    rebalance_update_time: float
    num_physical_experts: int
    num_logical_experts: int
    num_redundant_experts: int
    logical_count: torch.Tensor
    physical_to_logical_map_summary: Optional[Dict] = None
    logical_to_physical_map_summary: Optional[Dict] = None
    gpu_expert_stats: Optional[Dict] = None


class EPLBMetricsCollector:
    def __init__(self, labels: Dict[str, str]) -> None:
        from prometheus_client import Gauge, Histogram, Info
        
        self.labels = labels
        
        # 存储当前的logical_expert_replicas数据，用于自定义输出
        self._current_logical_expert_data = {}
        
        self.rebalance_time = Histogram(
            name="sglang:eplb_rebalance_time_seconds",
            documentation="Histogram of EPLB rebalance time in seconds",
            labelnames=list(labels.keys()) + ["stage"], 
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        )

        self.expert_tokens = Gauge(
            name="sglang:eplb_expert_tokens",
            documentation="Number of EPLB expert tokens",
            labelnames=list(labels.keys()) + ["layer_id", "expert_id"], 
            multiprocess_mode="mostrecent",
        )
        
        self.num_experts = Gauge(
            name="sglang:eplb_num_experts",
            documentation="Number of experts",
            labelnames=list(labels.keys()) + ["type"],
            multiprocess_mode="mostrecent",
        )
        
        self.load_stats = Gauge(
            name="sglang:eplb_load_stats",
            documentation="Expert load statistics",
            labelnames=list(labels.keys()) + ["metric"],
            multiprocess_mode="mostrecent",
        )
        
        self.gpu_expert_stats = Gauge(
            name="sglang:eplb_gpu_expert_stats",
            documentation="Expert statistics per GPU",
            labelnames=list(labels.keys()) + ["gpu_id", "layer_id", "metric"],
            multiprocess_mode="mostrecent",
        )
        
        self.logical_expert_replicas = Gauge(
            name="sglang:eplb_logical_expert_replicas",
            documentation="Number of physical expert replicas for each logical expert",
            labelnames=list(labels.keys()) + ["gpu_id", "layer_id", "logical_expert_id"],
            multiprocess_mode="mostrecent",
        )
    
    def _log_gauge(self, gauge, data: Union[int, float], extra_labels: Dict[str, str] = None) -> None:
        labels = self.labels.copy()
        if extra_labels:
            labels.update(extra_labels)
        gauge.labels(**labels).set(data)
    
    def generate_custom_metrics(self):
        """
        生成自定义的Prometheus格式metrics文本，包含所有metrics指标。
        对于logical_expert_replicas指标，使用我们存储的当前数据，过滤掉历史数据。
        对于其他指标，直接使用Prometheus客户端库生成的文本。
        """
        import time
        from prometheus_client import generate_latest, REGISTRY
        import logging
        
        # 获取日志记录器
        logger = logging.getLogger(__name__)
        
        # 调试：检查当前数据
        logger.info(f"generate_custom_metrics called, _current_logical_expert_data has {len(self._current_logical_expert_data)} entries")
        if self._current_logical_expert_data:
            logger.info(f"Sample entries: {list(self._current_logical_expert_data.items())[:3]}")
        
        # 从Prometheus客户端库生成所有指标的文本
        all_metrics = generate_latest(REGISTRY).decode('utf-8')
        
        # 调试：检查原始metrics中是否有这个指标
        has_original_metric = 'sglang:eplb_logical_expert_replicas' in all_metrics
        logger.info(f"Original metrics contains logical_expert_replicas: {has_original_metric}")
        
        # 分割指标文本为不同的指标块
        metrics_blocks = {}
        current_block = []
        current_name = None
        
        for line in all_metrics.split('\n'):
            if line.startswith('# HELP '):
                if current_name and current_block:
                    metrics_blocks[current_name] = current_block
                current_name = line.split(' ')[2]
                current_block = [line]
            elif line.strip():
                if current_block is not None:
                    current_block.append(line)
        
        # 添加最后一个块
        if current_name and current_block:
            metrics_blocks[current_name] = current_block
        
        # 替换logical_expert_replicas指标块
        expert_metrics_count = 0
        if 'sglang:eplb_logical_expert_replicas' in metrics_blocks:
            logger.info("Found existing logical_expert_replicas block, replacing it")
            # 保留HELP和TYPE行
            help_line = metrics_blocks['sglang:eplb_logical_expert_replicas'][0]
            type_line = metrics_blocks['sglang:eplb_logical_expert_replicas'][1]
            
            # 使用我们的当前数据生成新的指标行
            new_lines = [help_line, type_line]
            
            for (gpu_id, layer_id, logical_id), value in self._current_logical_expert_data.items():
                # 构建标签字符串，保持与其他指标一致的顺序
                label_parts = [
                    f'gpu_id="{gpu_id}"',
                    f'layer_id="{layer_id}"',
                    f'logical_expert_id="{logical_id}"'
                ]
                # 添加基础标签
                for label_name, label_value in self.labels.items():
                    label_parts.append(f'{label_name}="{label_value}"')
                labels_str = ','.join(label_parts)
                
                # 添加指标行（移除时间戳，让Prometheus自动处理）
                new_lines.append(f'sglang:eplb_logical_expert_replicas{{{labels_str}}} {float(value)}')
                expert_metrics_count += 1
            
            # 替换原始块（即使new_lines只有HELP和TYPE行）
            metrics_blocks['sglang:eplb_logical_expert_replicas'] = new_lines
        else:
            logger.info("No existing logical_expert_replicas block found, creating new one")
            # 如果原始输出中没有这个指标块，创建一个新的
            new_lines = [
                "# HELP sglang:eplb_logical_expert_replicas Number of physical expert replicas for each logical expert",
                "# TYPE sglang:eplb_logical_expert_replicas gauge"
            ]
            
            for (gpu_id, layer_id, logical_id), value in self._current_logical_expert_data.items():
                # 构建标签字符串，保持与其他指标一致的顺序
                label_parts = [
                    f'gpu_id="{gpu_id}"',
                    f'layer_id="{layer_id}"',
                    f'logical_expert_id="{logical_id}"'
                ]
                # 添加基础标签
                for label_name, label_value in self.labels.items():
                    label_parts.append(f'{label_name}="{label_value}"')
                labels_str = ','.join(label_parts)
                
                # 添加指标行（移除时间戳，让Prometheus自动处理）
                new_lines.append(f'sglang:eplb_logical_expert_replicas{{{labels_str}}} {float(value)}')
                expert_metrics_count += 1
            
            metrics_blocks['sglang:eplb_logical_expert_replicas'] = new_lines
        
        # 记录处理的指标数量
        total_metrics_count = sum(len(block) - 2 for block in metrics_blocks.values())  # 减去每个块的HELP和TYPE行
        logger.info(f"Generated custom metrics: {len(metrics_blocks)} metric types, "
                    f"{total_metrics_count} total data points, "
                    f"{expert_metrics_count} logical expert replica metrics")
        
        # 重新组合所有指标块
        output_lines = []
        for block_name, block_lines in metrics_blocks.items():
            output_lines.extend(block_lines)
        
        return '\n'.join(output_lines)
    
    def log_stats(self, stats: EPLBManagerStats) -> None:
        self.rebalance_time.labels(**self.labels, stage="total").observe(stats.rebalance_total_time)
        self.rebalance_time.labels(**self.labels, stage="compute").observe(stats.rebalance_compute_time)
        self.rebalance_time.labels(**self.labels, stage="update").observe(stats.rebalance_update_time)
        
        self.num_experts._metrics.clear()
        self._log_gauge(self.num_experts, stats.num_physical_experts, {"type": "physical"})
        self._log_gauge(self.num_experts, stats.num_logical_experts, {"type": "logical"})
        self._log_gauge(self.num_experts, stats.num_redundant_experts, {"type": "redundant"})
        
        # 检查logical_count是否为None
        if stats.logical_count is not None:
            mean_load = stats.logical_count.float().mean()
            std_load = stats.logical_count.float().std()
            cv = std_load / mean_load if mean_load > 0 else 0
            max_load = stats.logical_count.max().item()
            min_load = stats.logical_count.min().item()
            
            load_cv = float(cv) if isinstance(cv, torch.Tensor) else cv
            load_max = max_load
            load_min = min_load
            load_mean = float(mean_load) if isinstance(mean_load, torch.Tensor) else mean_load

            self.load_stats._metrics.clear()
            self._log_gauge(self.load_stats, load_cv, {"metric": "cv"})
            self._log_gauge(self.load_stats, load_max, {"metric": "max"})
            self._log_gauge(self.load_stats, load_min, {"metric": "min"})
            self._log_gauge(self.load_stats, load_mean, {"metric": "mean"})
        
            self.expert_tokens._metrics.clear()
            for layer_id in range(stats.logical_count.shape[0]):
                for logical_expert_id in range(stats.logical_count.shape[1]):
                    tokens_count = stats.logical_count[layer_id, logical_expert_id].item()
                    # each layer each expert has a different number of tokens
                    #  self.expert_tokens.labels(**self.labels, layer_id=str(layer_id), expert_id=str(logical_expert_id)).observe(tokens_count)
                    self._log_gauge(self.expert_tokens, tokens_count, {"layer_id": str(layer_id), "expert_id": str(logical_expert_id)})
            
        if stats.gpu_expert_stats:
            # 清空当前的logical_expert_data
            self._current_logical_expert_data = {}
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"log_stats called with gpu_expert_stats containing {len(stats.gpu_expert_stats)} GPUs")
            
            # 设置新数据
            for gpu_id, layer_stats in stats.gpu_expert_stats.items():
                for layer_id, metrics in layer_stats.items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            self.gpu_expert_stats.labels(
                                **self.labels, 
                                gpu_id=str(gpu_id), 
                                layer_id=str(layer_id), 
                                metric=metric_name
                            ).set(value)
                    
                    if "logical_expert_counts" in metrics:
                        for logical_id, local_count in metrics["logical_expert_counts"].items():
                            # 更新用于自定义输出的数据结构
                            self._current_logical_expert_data[(str(gpu_id), str(layer_id), str(logical_id))] = local_count
                            
                            # 不再直接更新Prometheus指标，只通过generate_custom_metrics输出
                            # self.logical_expert_replicas.labels(
                            #     **self.labels,
                            #     gpu_id=str(gpu_id),
                            #     layer_id=str(layer_id),
                            #     logical_expert_id=str(logical_id)
                            # ).set(local_count)
            
            logger.info(f"Updated _current_logical_expert_data with {len(self._current_logical_expert_data)} entries")
            if self._current_logical_expert_data:
                logger.info(f"Sample entries: {list(self._current_logical_expert_data.items())[:3]}")
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("log_stats called but stats.gpu_expert_stats is None or empty")


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    cache_hit_rate: float = 0.0
    spec_accept_length: float = 0.0
    avg_request_queue_latency: float = 0.0


class SchedulerMetricsCollector:

    def __init__(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Gauge, Histogram

        self.labels = labels
        self.last_log_time = time.time()

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generation throughput (token/s).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.spec_accept_length = Gauge(
            name="sglang:spec_accept_length",
            documentation="The average acceptance length of speculative decoding.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.avg_request_queue_latency = Gauge(
            name="sglang:avg_request_queue_latency",
            documentation="The average request queue latency for the last batch of requests in seconds.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def log_stats(self, stats: SchedulerStats) -> None:
        self._log_gauge(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.num_queue_reqs, stats.num_queue_reqs)
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        self._log_gauge(self.avg_request_queue_latency, stats.avg_request_queue_latency)
        self.last_log_time = time.time()


class TokenizerMetricsCollector:
    def __init__(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels

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

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=[
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
            ],
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=[
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
            ],
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=[
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
            ],
        )

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
    ):
        self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
        if cached_tokens > 0:
            self.cached_tokens_total.labels(**self.labels).inc(cached_tokens)
        self.num_requests_total.labels(**self.labels).inc(1)
        self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)

    def observe_time_to_first_token(self, value: float):
        self.histogram_time_to_first_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(self, internval: float, num_new_tokens: int):
        adjusted_interval = internval / num_new_tokens

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
        his._sum.inc(internval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break

# 全局实例，供其他模块访问
eplb_metrics_collector = None

def create_eplb_metrics_collector(labels: Dict[str, str]) -> EPLBMetricsCollector:
    """创建并设置全局EPLB指标收集器实例"""
    global eplb_metrics_collector
    eplb_metrics_collector = EPLBMetricsCollector(labels)
    return eplb_metrics_collector
