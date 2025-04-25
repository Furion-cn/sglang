import hashlib
import torch
import os
from typing import Union
import zmq
import logging
import socket
import threading
import time
import uuid
import nvtx
from typing import List, Dict
from dataclasses import dataclass
import bisect

from sglang.srt.managers.io_struct import PrefilledReqInput, KVTransferFetch, KVTransferAck
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.pd_disaggregation_controller import PD_DISAGGREGATION_PORT
from sglang.srt.server_args import ServerArgs
from sglang.srt.mem_cache.radix_cache import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.utils import get_zmq_socket

logger = logging.getLogger(__name__)


class TransferEngine:
    """Handles the transfer of data using mooncake_transfer_engine and ZeroMQ."""

    def __init__(self,
                 local_host: str,
                 metadata_server: str,
                 device_name: str):
        try:
            import engine
            import etcd3
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run engine.") from e

        # delete all prefix kv cache from etcd in order to exceed the space.
        parts = metadata_server.split(':')
        addr_prefix = '_'.join(local_host.split('_')[:-1])
        prefixes = [
            'mooncake/rpc_meta/'+addr_prefix,
            'mooncake/ram/'+addr_prefix,
        ]
        for prefix in prefixes:
            etcd3.client(host=parts[0], port=parts[1]).delete_prefix(prefix)

        self.engine = engine.TransferEngine()
        self.engine.initialize(local_host, metadata_server,
                            "rdma", device_name)

        logger.info(f"TransferEngine initialized.")

    def register_memory(self, buffer: int, length: int) -> int:
        """Register a memory buffer."""
        return self.engine.register_memory(buffer, length)

    def transfer_sync(self, remote_url: str, buffer: int, peer_buffer_address: int,
                      length: int) -> int:
        """Synchronously transfer data to the specified address."""
        ret = self.engine.transfer_sync_write(remote_url, buffer,
                                              peer_buffer_address, length)
        if ret < 0:
            logger.error(f"Transfer Return Error: {ret}")
            raise Exception(f"Transfer Return Error: {ret}")
        return ret


class KVBufferBlock:
    def __init__(self, offset: int, capacity: int, length: int = None):
        self.offset = offset
        # Total allocated capacity (may be aligned to block size)
        self.capacity = capacity
        # If length is not specified, use 0 to indicate it's free
        self.length = length if length is not None else 0  # Actual used length


@dataclass
class KVBufferStats:
    capacity_size: int
    used_size: int
    available_size: int
    available_block_sizes: List[int]


class KVBuffer:
    def __init__(self, cache: torch.Tensor, block_sizes: List[int] = None):
        self.cache = cache
        self.blocks: Dict[int, KVBufferBlock] = {
            0: KVBufferBlock(0, cache.shape[0])}
        self.free_slots: List[int] = [0]

        # Block size strategy
        self.block_sizes = block_sizes
        if self.block_sizes:
            self.block_sizes.sort()  # Ensure block sizes are sorted

        logger.debug(
            f"KVBuffer initialized with block sizes: {self.block_sizes}")

    def _align_to_block_size(self, length: int) -> int:
        """Align length to predefined block sizes"""
        if not self.block_sizes:
            return length

        # Find the smallest block size that is greater than or equal to length
        for size in self.block_sizes:
            if size >= length:
                return size
        # If larger than the maximum block size, round up to a multiple of the largest block
        largest_block = self.block_sizes[-1]
        return ((length + largest_block - 1) // largest_block) * largest_block

    def set_item(self, item: torch.Tensor, non_blocking: bool = False) -> int:
        assert item.shape[1:] == self.cache.shape[1:], \
            f"item shape {item.shape} does not match cache shape {self.cache.shape}"
        offset = self.allocate(item.shape[0])
        self.cache[offset:offset + item.shape[0]].copy_(item, non_blocking=non_blocking)
        return offset

    def get_item(self, offset: int) -> torch.Tensor:
        # Use the length to get the exact tensor that was set
        return self.cache[offset:offset + self.blocks[offset].length]

    def allocate(self, length: int) -> int:
        # Align request length to block size
        aligned_capacity = self._align_to_block_size(length)

        offset = self._allocate(aligned_capacity, length)
        if offset < 0:
            self._clean_fragment()
            offset = self._allocate(aligned_capacity, length)
        if offset < 0:
            raise Exception(
                f"No enough free space for length {length} (aligned to {aligned_capacity}, stats: {self.stats()})")
        return offset

    def _allocate(self, capacity: int, length: int = None) -> int:
        if len(self.free_slots) == 0:
            raise Exception("No free slot")

        # First try to find the best fitting block (best fit)
        best_fit_idx = -1
        best_fit_size = float('inf')

        for i, offset in enumerate(self.free_slots):
            block = self.blocks[offset]
            if block.capacity < capacity:
                continue

            # Found a block that's large enough, calculate wasted space
            wasted_space = block.capacity - capacity
            if wasted_space < best_fit_size:
                best_fit_idx = i
                best_fit_size = wasted_space

                # If perfect match found, use immediately
                if wasted_space == 0:
                    break

        if best_fit_idx == -1:
            return -1  # No suitable block found

        # Use the best fitting block
        offset = self.free_slots.pop(best_fit_idx)
        block = self.blocks[offset]

        if block.capacity > capacity:
            # Create a new block with the requested capacity and length
            allocated_block = KVBufferBlock(block.offset, capacity, length)
            # Create a remaining block for the unused space
            remaining_block = KVBufferBlock(
                block.offset + capacity, block.capacity - capacity)
            self.blocks[block.offset] = allocated_block
            self.blocks[remaining_block.offset] = remaining_block
            self.free(remaining_block.offset)
        elif length is not None:
            # Block size matches exactly, just update the length
            block.length = length

        return block.offset

    def free(self, offset: int):
        # Check if the offset exists in blocks
        if offset not in self.blocks:
            raise ValueError(
                f"Cannot free: memory block at offset {offset} does not exist")

        # Check if the block is already freed
        if offset in self.free_slots:
            raise ValueError(
                f"Memory block at offset {offset} is already freed")

        # Reset the block length to 0 to indicate it's free
        self.blocks[offset].length = 0
        bisect.insort(self.free_slots, offset)
        self._merge_adjacent_blocks(offset)

    def _merge_adjacent_blocks(self, offset: int):
        if len(self.free_slots) <= 1:
            return

        cur_index = max(0, self.free_slots.index(offset) - 1)
        for _ in range(2):
            if cur_index < 0 or cur_index+1 >= len(self.free_slots):
                continue
            curr_offset = self.free_slots[cur_index]
            next_offset = self.free_slots[cur_index+1]
            if curr_offset + self.blocks[curr_offset].capacity != next_offset:
                cur_index += 1
                continue
            # merge blocks
            self.blocks[curr_offset].capacity += self.blocks[next_offset].capacity
            del self.blocks[next_offset]
            self.free_slots.pop(cur_index+1)

    # Merge fragmented blocks
    def _clean_fragment(self):
        if len(self.free_slots) <= 1:
            return

        # No need to sort as free_slots is always kept sorted by bisect.insort
        # Keep the first slot as a starting point
        merged_slots = [self.free_slots[0]]
        current_offset = self.free_slots[0]
        current_capacity = self.blocks[current_offset].capacity

        for i in range(1, len(self.free_slots)):
            next_offset = self.free_slots[i]
            next_capacity = self.blocks[next_offset].capacity

            # Check if adjacent
            if current_offset + current_capacity == next_offset:
                # Merge blocks
                current_capacity += next_capacity
                self.blocks[current_offset] = KVBufferBlock(
                    current_offset, current_capacity)
                # Delete the merged block
                del self.blocks[next_offset]
            else:
                # Not adjacent, add to merged list and update current pointer
                merged_slots.append(next_offset)
                current_offset = next_offset
                current_capacity = next_capacity

        # Replace the original list with the merged list
        self.free_slots = merged_slots

    def data_ptr(self, offset: int) -> int:
        return self.cache.data_ptr() + offset * self.element_size()

    def element_size(self) -> int:
        return self.cache.stride(0) * self.cache.element_size()

    def stats(self) -> KVBufferStats:
        return KVBufferStats(
            capacity_size=self.cache.shape[0],
            used_size=sum(
                block.capacity for block in self.blocks.values() if block.length > 0),
            available_size=sum(
                block.capacity for block in self.blocks.values() if block.length == 0),
            available_block_sizes=[
                block.capacity for block in self.blocks.values() if block.length == 0]
        )


KV_TRANSFER_AGENT_PORT = 19000


class KVTransferAgent:
    def __init__(self,
                 server_args: ServerArgs,
                 req_to_token_pool: ReqToTokenPool = None,
                 token_to_kv_pool_allocator: TokenToKVPoolAllocator = None,
                 layer_num: int = 0,
                 tp_rank: int = 0,
                 kv_cache_capacity: int = 0,
                 device: str = "cpu:0"):
        self.layer_num = layer_num
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.role = server_args.kv_transfer_config.role
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_kv_transfer_ct = {}
        self.req_kv_transfer_results = {}
        self.req_kv_transfer_lock = threading.Lock()

        self.attn_tp_rank, self.attn_tp_size, _ = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                server_args.tp_size,
                server_args.dp_size,
            )
        )

        tp_size_per_node = server_args.tp_size // server_args.nnodes
        hostname = os.environ.get(
            "KV_TRANSFER_AGENT_HOSTNAME", socket.gethostname())
        self.addr = f"{hostname}_{self.tp_rank % tp_size_per_node}_{uuid.uuid4()}"
        logger.debug(f"KVTransferAgent addr: {self.addr}")

        if server_args.nnodes == 1 and server_args.dist_init_addr is None:
            dist_init_host = "127.0.0.1"
        else:
            dist_init_host, _ = server_args.dist_init_addr.split(":")

        context = zmq.Context(2)

        if self.role == "decode":
            self.recv_from_pd_disagg_controller = get_zmq_socket(
                context, zmq.PULL, f"tcp://{dist_init_host}:{PD_DISAGGREGATION_PORT+self.tp_rank+1}", False)
            self.send_to_pd_disagg_controller = get_zmq_socket(
                context, zmq.PUSH, f"tcp://{server_args.kv_transfer_config.prefill_dist_init_host}:{PD_DISAGGREGATION_PORT}", False)
        else:
            self.recv_from_pd_disagg_controller = get_zmq_socket(
                context, zmq.PULL, f"tcp://{dist_init_host}:{PD_DISAGGREGATION_PORT+self.tp_rank+1}", False)
            self.send_to_pd_disagg_controller = get_zmq_socket(
                context, zmq.PUSH, f"tcp://{server_args.kv_transfer_config.decode_dist_init_host}:{PD_DISAGGREGATION_PORT}", False)

        self.device = device

        cache = torch.zeros(
            [kv_cache_capacity, self.layer_num, 1, 576], dtype=torch.bfloat16, device=self.device, pin_memory=True)
        self.engine = TransferEngine(
            self.addr, server_args.kv_transfer_config.transfer_engine_metadata_server, server_args.kv_transfer_config.transfer_engine_rdma_device)
        self.engine.register_memory(cache.data_ptr(), cache.nbytes)
        self.kv_buffer = KVBuffer(
            cache, block_sizes=[2**i for i in range(3, 14)])
        self.req_to_kv_buffer_offset = {}

    @nvtx.annotate("KVTransferAgent.set_kv_buffer", color="red")
    def set_kv_buffer(self, req: Req):
        if self.attn_tp_rank != 0:
            return 0
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        
        nvtx.push_range("KVTransferAgent.set_kv_buffer.stack_and_permute")
        kv_cache = torch.stack(
            [self.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(i)[kv_indices]
             for i in range(self.layer_num)]
        ).permute(1, 0, 2, 3).contiguous().to(self.device, non_blocking=True)
        nvtx.pop_range()
        
        nvtx.push_range("KVTransferAgent.set_kv_buffer.set_item")
        offset = self.kv_buffer.set_item(kv_cache, non_blocking=True)
        nvtx.pop_range()
        
        self.req_to_kv_buffer_offset[req.rid] = offset

    @nvtx.annotate("KVTransferAgent.get_kv_buffer", color="red")
    def get_kv_buffer(self, req_list: List[Req]) -> dict[str, torch.Tensor]:
        if not req_list:
            return {}

        # group requests by source address
        requests_by_src = {}
        for req in req_list:
            src_addr = req.kv_transfer_src_addr
            requests_by_src.setdefault(src_addr, []).append(req)

        # get kv buffer from same source address
        results = {}
        for src_reqs in requests_by_src.values():
            results.update(self._get_kv_buffer_from_same_src(src_reqs))
        return results

    @nvtx.annotate("KVTransferAgent._get_kv_buffer_from_same_src", color="red")
    def _get_kv_buffer_from_same_src(self, req_list: List[Req]) -> dict[str, torch.Tensor]:
        if len(req_list) == 0:
            return {}
        res = {}
        offsets = []
        try:
            start_time = time.time()
            dst_ptrs = []
            for req in req_list:
                offset = self.kv_buffer.allocate(len(req.origin_input_ids))
                dst_ptrs.append(self.kv_buffer.data_ptr(offset))
                offsets.append(offset)

            allocated_time = time.time()

            kv_transfer_fetch = KVTransferFetch(
                rids=[req.rid for req in req_list],
                src_addr=req_list[0].kv_transfer_src_addr,
                src_rank=req_list[0].kv_transfer_src_rank,
                dst_addr=self.addr,
                dst_rank=self.tp_rank,
                dst_ptrs=dst_ptrs,
                fetch_ct=self.attn_tp_size,
            )

            self._complete_kv_transfer(kv_transfer_fetch)
            transfered_time = time.time()
            transfered_bytes = 0
            for i, req in enumerate(req_list):
                res[req.rid] = self.kv_buffer.get_item(
                    offsets[i]).permute(1, 0, 2, 3)
                transfered_bytes += res[req.rid].nbytes

            loaded_time = time.time()

            logger.debug(
                f"[KVTransferAgent] Received kv cache ({transfered_bytes / 1024 / 1024} MB) \
                    allocated: {allocated_time - start_time} seconds \
                    transfered: {transfered_time - allocated_time} seconds \
                    loaded: {loaded_time - transfered_time} seconds \
                    total: {time.time() - start_time} seconds \
                    bandwidth: {transfered_bytes/1024/1024/1024 / (time.time() - start_time)} GB/s \
                    kv_buffer_stats: {self.kv_buffer.stats()}")

        except Exception as e:
            logger.error(f"[KVTransferAgent] Get batch kv buffer failed: {e}")

            for req in req_list:
                req.finished_reason = FINISH_ABORT(
                    message=f"Get batch kv buffer failed: {e}")

            return {}
        finally:
            for offset in offsets:
                self.kv_buffer.free(offset)
            return res

    @nvtx.annotate("KVTransferAgent._complete_kv_transfer", color="red")
    def _complete_kv_transfer(self, kv_transfer_fetch: KVTransferFetch, timeout: int = 60):
        """Complete kv transfer.
        If timeout, raise an exception. Default timeout is 60 seconds.
        """
        # Create an event for this group of requests
        event = threading.Event()
        batch_id = hashlib.md5(
            str(kv_transfer_fetch.rids).encode()).hexdigest()

        with self.req_kv_transfer_lock:
            self.req_kv_transfer_results[batch_id] = {
                'event': event, 'ack': None}

        self.send_to_pd_disagg_controller.send_pyobj(kv_transfer_fetch)

        logger.debug(
            f"[KVTransferAgent] Waiting for kv transfer to be done")

        # Wait for event to be triggered or timeout
        if not event.wait(timeout):
            with self.req_kv_transfer_lock:
                del self.req_kv_transfer_results[batch_id]
            raise Exception("KV transfer timeout")

        # Check results and errors
        with self.req_kv_transfer_lock:
            ack = self.req_kv_transfer_results[batch_id]['ack']
            if ack is None:
                raise Exception(
                    f"KV transfer for {kv_transfer_fetch.rids} incomplete despite event trigger")
            if ack.error_message is not None:
                raise Exception(
                    f"KV transfer {kv_transfer_fetch.rids} failed: {ack.error_message}")
            del self.req_kv_transfer_results[batch_id]

    def dispatch_prefilled_req(self, req: Req):
        if self.role == "decode":
            return
        if self.attn_tp_rank != 0:
            return

        logger.debug(
            f"[KVTransferAgent] Dispatch prefilled request {req.rid}")

        self.send_to_pd_disagg_controller.send_pyobj(PrefilledReqInput(
            rid=req.rid,
            mm_inputs=None,
            input_text=req.origin_input_text,
            input_ids=req.origin_input_ids,
            sampling_params=req.sampling_params,
            return_logprob=req.return_logprob,
            logprob_start_len=req.logprob_start_len,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            stream=req.stream,
            output_ids=req.output_ids,
            kv_transfer_src_addr=self.addr,
            kv_transfer_src_rank=self.tp_rank,
        ))

        logger.debug(
            f"[KVTransferAgent] Dispatched prefilled request {req.rid}")

    def _handle_kv_transfer_fetch(self, req: KVTransferFetch):
        transfered_bytes = 0
        try:
            start_time = time.time()

            for i, rid in enumerate(req.rids):
                offset = self.req_to_kv_buffer_offset[rid]
                if offset is None:
                    raise Exception(
                        f"KV transfer fetch request {rid} not found")

                if rid not in self.req_kv_transfer_ct:  # first time fetch
                    self.req_kv_transfer_ct[rid] = req.fetch_ct

                kv_cache = self.kv_buffer.get_item(offset)
                self.engine.transfer_sync(
                    req.dst_addr, self.kv_buffer.data_ptr(offset), req.dst_ptrs[i], kv_cache.nbytes)

                self.req_kv_transfer_ct[rid] -= 1

                transfered_bytes += kv_cache.nbytes
            logger.debug(
                f"[KVTransferAgent] Transferred kv cache to RANK_{req.dst_rank} ({transfered_bytes / 1024 / 1024} MB) in {time.time() - start_time} seconds, bandwidth: {transfered_bytes/1024/1024/1024 / (time.time() - start_time)} GB/s, kv_buffer_stats: {self.kv_buffer.stats()}")

            ack_error_message = None
        except Exception as e:
            logger.error(f"[KVTransferAgent] KV transfer failed: {e}")
            ack_error_message = str(e)

        # send ack to remote addr
        self.send_to_pd_disagg_controller.send_pyobj(
            KVTransferAck(req.rids, req.dst_addr, req.dst_rank, ack_error_message))

        if ack_error_message is not None:
            return

        # free buffer
        for rid in req.rids:
            if self.req_kv_transfer_ct[rid] > 0:
                return
            self.kv_buffer.free(self.req_to_kv_buffer_offset[rid])
            del self.req_to_kv_buffer_offset[rid]

    def _handle_kv_transfer_ack(self, kv_transfer_ack: KVTransferAck):
        if len(kv_transfer_ack.rids) == 0:
            return

        batch_id = hashlib.md5(
            str(kv_transfer_ack.rids).encode()).hexdigest()

        with self.req_kv_transfer_lock:
            if batch_id not in self.req_kv_transfer_results:
                raise Exception(
                    f"KV transfer ack request {kv_transfer_ack.rids} not in req_kv_transfer_results")

            # Store the result
            self.req_kv_transfer_results[batch_id]['ack'] = kv_transfer_ack
            # trigger event for the first request in the batch
            self.req_kv_transfer_results[batch_id]['event'].set()

    def event_loop(self):
        while True:
            recv_obj = self.recv_from_pd_disagg_controller.recv_pyobj()
            if isinstance(recv_obj, KVTransferFetch):
                self._handle_kv_transfer_fetch(recv_obj)
            elif isinstance(recv_obj, KVTransferAck):
                self._handle_kv_transfer_ack(recv_obj)
            else:
                raise ValueError(
                    f"[KVTransferAgent] Unknown message type: {type(recv_obj)}")
