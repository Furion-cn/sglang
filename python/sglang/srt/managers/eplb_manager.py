import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.managers.io_struct import (
    EplbRebalanceReqInput,
    UpdateExpertLocationReqInput,
)
from sglang.srt.metrics.collector import EPLBManagerStats, EPLBMetricsCollector
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, server_args: ServerArgs):
        super().__init__()
        self._server_args = server_args
        self._expert_distribution_storage = ExpertDistributionStorage(
            dir_data=Path(self._server_args.eplb_storage_dir)
            / "expert_distribution_storage"
        )
        self._metrics_collector = EPLBMetricsCollector(
            labels={"model": server_args.served_model_name, "node_rank": str(server_args.node_rank)}
        )
        self._expert_location_metadata = None

    def bind(self, tokenizer_manager: "TokenizerManager"):
        self._tokenizer_manager = tokenizer_manager
        self._expert_distribution_storage.bind(tokenizer_manager)

    async def handle_loop(self):
        await self._expert_distribution_storage.start()
        while True:
            sleep_time = self._server_args.eplb_rebalance_period or 1000000000
            logger.info(
                f"EPLBManager: Sleep {sleep_time} seconds before next automatic rebalancing"
            )
            await asyncio.sleep(sleep_time)
            await self.rebalance(EplbRebalanceReqInput())

    async def rebalance(self, obj: EplbRebalanceReqInput):
        start_time = time.time()
        
        await self.save_expert_distribution()
        
        compute_start = time.time()
        expert_location_metadata = self.compute_expert_location_metadata(
            debug_use_random_stat=obj.debug_use_random_stat
        )
        compute_time = time.time() - compute_start
        
        update_start = time.time()
        await self._tokenizer_manager.update_expert_location(
            UpdateExpertLocationReqInput(
                expert_location_metadata=expert_location_metadata
            )
        )
        update_time = time.time() - update_start
        
        total_time = time.time() - start_time
        
        self._expert_location_metadata = expert_location_metadata
        
        self._collect_and_report_metrics(
            expert_location_metadata, 
            total_time, 
            compute_time, 
            update_time
        )

    async def save_expert_distribution(self):
        await self._expert_distribution_storage.save_current()

    def compute_expert_location_metadata(self, debug_use_random_stat: bool = False):
        snapshot = self._expert_distribution_storage.get_last_snapshot()
        if snapshot is None:
            metadata = ExpertLocationMetadata.init_trivial(self._server_args)
            logger.info("EPLBManager: Initial trivial expert distribution (no snapshot available)")
            self._log_expert_maps(metadata)
            return metadata

        if debug_use_random_stat:
            logger.warning(
                "EPLBManager.compute_expert_location_metadata use random stat for debugging."
            )
            original_logical_count = torch.tensor(snapshot["logical_count"])
            snapshot = {
                "logical_count": torch.randint_like(original_logical_count, high=100000)
            }

        metadata = ExpertLocationMetadata.init_by_eplb(self._server_args, **snapshot)
        logger.info("EPLBManager: Expert distribution computed by EPLB")
        self._log_expert_maps(metadata)
        return metadata
        
    def _log_expert_maps(self, metadata: ExpertLocationMetadata):
        logger.info(
            f"EPLBManager: Expert distribution - "
            f"logical_experts={metadata.num_logical_experts}, "
            f"physical_experts={metadata.num_physical_experts}, "
            f"redundant_experts={metadata.num_physical_experts - metadata.num_logical_experts}"
        )
        
        physical_expert_counts = {}
        for layer_id in range(metadata.num_layers):
            for rank_id in range(metadata.num_physical_experts // metadata.num_local_physical_experts):
                key = f"layer_{layer_id}_rank_{rank_id}"
                if key not in physical_expert_counts:
                    physical_expert_counts[key] = 0
                physical_expert_counts[key] += metadata.num_local_physical_experts
        
        replica_counts = {}
        for layer_id in range(metadata.num_layers):
            for logical_id in range(metadata.num_logical_experts):
                replicas = metadata.logical_to_all_physical_map_num_valid[layer_id, logical_id].item()
                if replicas not in replica_counts:
                    replica_counts[replicas] = 0
                replica_counts[replicas] += 1
        
        # logger.info(f"EPLBManager: Expert replica distribution: {json.dumps(replica_counts)}")
        # p2l_map = {f"rank_{i}": row.tolist() for i, row in enumerate(metadata.physical_to_logical_map)}
        # logger.info(f"EPLBManager: Expert physical_to_logical_map: {json.dumps(p2l_map)}")
        
        # l2p_map = {f"expert_{i}": row.tolist() for i, row in enumerate(metadata.logical_to_all_physical_map)}
        # logger.info(f"EPLBManager: Expert logical_to_all_physical_map: {json.dumps(l2p_map)}")
    
    def _collect_and_report_metrics(
        self, 
        metadata: ExpertLocationMetadata, 
        total_time: float, 
        compute_time: float, 
        update_time: float
    ):
        snapshot = self._expert_distribution_storage.get_last_snapshot()
                
        p2l_map_summary = self._create_map_summary(metadata.physical_to_logical_map)
        l2p_map_summary = self._create_map_summary(metadata.logical_to_all_physical_map)
        
        gpu_expert_stats = self._compute_gpu_expert_stats(metadata)
        
        stats = EPLBManagerStats(
            rebalance_total_time=total_time,
            rebalance_compute_time=compute_time,
            rebalance_update_time=update_time,
            num_physical_experts=metadata.num_physical_experts,
            num_logical_experts=metadata.num_logical_experts,
            num_redundant_experts=metadata.num_physical_experts - metadata.num_logical_experts,
            logical_count=self._compute_load_balance_metrics(snapshot),
            physical_to_logical_map_summary=p2l_map_summary,
            logical_to_physical_map_summary=l2p_map_summary,
            gpu_expert_stats=gpu_expert_stats
        )
        
        self._metrics_collector.log_stats(stats)
        
        logger.info(f"EPLBManager: Rebalance metrics - "
                   f"time={total_time:.2f}s, "
                   f"experts={metadata.num_logical_experts}/{metadata.num_physical_experts}, "
                   f"load_cv={load_stats.get('load_cv', 0.0):.4f}")
    
    def _compute_gpu_expert_stats(self, metadata: ExpertLocationMetadata):
        gpu_expert_stats = {}
        
        num_gpus = metadata.ep_size
        num_layers = metadata.num_layers
        
        for gpu_id in range(num_gpus):
            gpu_expert_stats[gpu_id] = {}
            
            for layer_id in range(num_layers):
                start_idx = gpu_id * metadata.num_local_physical_experts
                end_idx = start_idx + metadata.num_local_physical_experts
                
                physical_to_logical = metadata.physical_to_logical_map[layer_id, start_idx:end_idx]
                
                unique_logical_experts = torch.unique(physical_to_logical)
                
                logical_expert_counts = {}
                for logical_id in unique_logical_experts.tolist():
                    if logical_id >= 0: 
                        count = torch.sum(physical_to_logical == logical_id).item()
                        logical_expert_counts[logical_id] = count
                    
                gpu_expert_stats[gpu_id][layer_id] = {
                    "num_physical_experts": metadata.num_local_physical_experts,
                    "num_unique_logical_experts": len(unique_logical_experts),
                    "utilization_ratio": len(unique_logical_experts) / metadata.num_local_physical_experts,
                    "logical_expert_counts": logical_expert_counts
                }
        
        return gpu_expert_stats
    
    def _compute_load_balance_metrics(self, snapshot: Dict[str, Any]) -> torch.Tensor:
        if snapshot is None:
            return {}
            
        logical_count = snapshot["logical_count"]
        
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)


        return logical_count
        # mean_load = logical_count.float().mean()
        # std_load = logical_count.float().std()
        # cv = std_load / mean_load if mean_load > 0 else 0
        # max_load = logical_count.max().item()
        # min_load = logical_count.min().item()
        
        # return {
        #     "load_cv": float(cv) if isinstance(cv, torch.Tensor) else cv,
        #     "max_load": max_load,
        #     "min_load": min_load,
        #     "mean_load": float(mean_load) if isinstance(mean_load, torch.Tensor) else mean_load
        # }
    
    def _create_map_summary(self, tensor_map):
        if tensor_map is None:
            return {}
            
        map_list = tensor_map.tolist()
        
        summary = {}
        for i, row in enumerate(map_list):
            summary[f"row_{i}"] = row
                
        return summary
