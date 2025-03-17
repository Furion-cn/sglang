from enum import Enum, auto
import logging
import psutil
from typing import Dict, Optional, List
import threading
import time
import numpy as np

logger = logging.getLogger(__name__)

class MemoryLimitMethod(Enum):
    """Memory Limit method."""
    RADIO_OF_MEMORY = auto()
    BYTES_OF_MEMORY = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid memory limit method: {method}") from exc

class HostMemoryManager(): 
    """Handle host memory of kv cache for PD disaggregation"""
    def __init__(
        self,
        limit_method: MemoryLimitMethod,
        limit_value: float, # 如果是 radio，则为 0-1 之间的值；如果是 bytes，则为字节数
        reserve_memory_bytes: int = 10 * 1024 * 1024 * 1024, # 默认保留 10GB 内存
        enable_manager: bool = True,
        memory_monitor_interval: int = 10, # 监控内存的间隔，默认 10 秒
        pre_allocate: bool = True, # 是否预分配内存
    ):
        self.enable_manager = enable_manager
        self._lock = threading.Lock()
        self.pre_allocate = pre_allocate

        if self.enable_manager:
            if not isinstance(limit_method, MemoryLimitMethod):
                raise TypeError(f"limit_method must be an instance of MemoryLimitMethod, got {type(limit_method)}")
            if limit_method == MemoryLimitMethod.RADIO_OF_MEMORY:
                if not 0 < limit_value <= 1:
                    raise ValueError(f"When using RADIO_OF_MEMORY, limit_value must be between 0 and 1, got {limit_value}")
            elif limit_method == MemoryLimitMethod.BYTES_OF_MEMORY:
                if limit_value <= 0:
                    raise ValueError(f"When using BYTES_OF_MEMORY, limit_value must be positive, got {limit_value}")

            self.limit_method = limit_method
            self.limit_value = limit_value
            self.reserve_memory_bytes = reserve_memory_bytes
            self.allocated_memory = {}  # rid -> bytes
            self.total_allocated_memory = 0
            self._memory_blocks = {}  # rid -> numpy array
            self._data_offsets = {}  # rid -> (offset, size)  记录每个请求的数据偏移量和大小
            self._calculate_memory_limit()
            if memory_monitor_interval > 0:
                self.start_memory_monitor(memory_monitor_interval)
                
            logger.info(f"HostMemoryManager initialized with limit method: {limit_method}, "
                f"limit value: {limit_value}, pre_allocate: {pre_allocate}")
        else:
            logger.info(f"HostMemoryManager is disabled.")

    def _calculate_memory_limit(self):
        mem_info = psutil.virtual_memory()
        total_memory = mem_info.total
        available_memory = mem_info.available
        
        if self.limit_method == MemoryLimitMethod.RADIO_OF_MEMORY:
            # 使用总内存的比例，但不超过当前可用内存。
            # 同时保留一部分内存保证稳定性
            calculated_memory = int(total_memory * self.limit_value)
            self.available_memory_bytes = min(calculated_memory, available_memory) - self.reserve_memory_bytes
        else: 
            # hard code 内存字节数，但不能超过当前可用内存
            # 同时保留一部分内存保证稳定性
            self.available_memory_bytes = min(self.limit_value, available_memory) - self.reserve_memory_bytes
        
        self.available_memory_bytes = max(0, self.available_memory_bytes)
        
        logger.debug(f"Memory limit calculated: total={total_memory}, available={available_memory}, "
                    f"limit={self.available_memory_bytes}")

    def can_allocate(self, size_bytes: int) -> bool:
        """check if we can allocate the memory
        
        Args:
            size_bytes: needed size in bytes
            
        Returns:
            bool: True if we can allocate the memory, False otherwise
        """
        if not self.enable_manager:
            return True
            
        with self._lock:
            current_free_memory = psutil.virtual_memory().available
            
            remaining_limit = self.available_memory_bytes - self.total_allocated_memory
            
            can_alloc = (size_bytes <= remaining_limit) and (size_bytes <= current_free_memory - self.reserve_memory_bytes)
            
            if not can_alloc:
                logger.warning(f"can not allocate {size_bytes} bytes memory。"
                              f"remaining limit: {remaining_limit}, "
                              f"current free memory: {current_free_memory}, "
                              f"total allocated memory: {self.total_allocated_memory}")
            
            return can_alloc

    def allocate(self, rid: str, size_bytes: int) -> bool:
        """try to allocate the memory for a request
        
        Args:
            rid: request id
            size_bytes: needed size in bytes
            
        Returns:
            bool: True if we can allocate the memory, False otherwise
        """
        if not self.enable_manager:
            return True
            
        with self._lock:
            if not self.can_allocate(size_bytes):
                return False
                
            # 释放旧的内存分配
            if rid in self.allocated_memory:
                old_size = self.allocated_memory[rid]
                self.total_allocated_memory -= old_size
                if rid in self._memory_blocks:
                    del self._memory_blocks[rid]
            
            # 预分配内存
            if self.pre_allocate:
                try:
                    memory_block = np.zeros(size_bytes, dtype=np.uint8)
                    self._memory_blocks[rid] = memory_block
                except MemoryError:
                    logger.error(f"Failed to pre-allocate {size_bytes} bytes for {rid}, system may be out of memory")
                    return False
            
            self.allocated_memory[rid] = size_bytes
            self.total_allocated_memory += size_bytes
            
            logger.debug(f"for {rid} allocated {size_bytes} bytes memory, "
                        f"total allocated memory: {self.total_allocated_memory}")
            
            return True

    def free(self, rid: str) -> int:
        """free the memory for a request
        
        Args:
            rid: request id
            
        Returns:
            int: freed size in bytes
        """
        if not self.enable_manager:
            return 0
            
        with self._lock:
            if rid not in self.allocated_memory:
                return 0
                
            freed_bytes = self.allocated_memory[rid]
            self.total_allocated_memory -= freed_bytes
            del self.allocated_memory[rid]
            
            # 释放预分配的内存块
            if rid in self._memory_blocks:
                del self._memory_blocks[rid]
            
            # 清除数据偏移量记录
            if rid in self._data_offsets:
                del self._data_offsets[rid]
            
            logger.debug(f"free {rid} 的 {freed_bytes} bytes memory, "
                        f"total allocated memory: {self.total_allocated_memory}")
            
            return freed_bytes
    
    def free_batch(self, rids: List[str]) -> int:
        """batch free memory for requests
        
        Args:
            rids: request ids
            
        Returns:
            int: total freed size in bytes
        """
        total_freed = 0
        for rid in rids:
            total_freed += self.free(rid)
        return total_freed

    def store_data(self, rid: str, data: bytes, offset: int = 0) -> bool:
        """store data to memory block
        
        Args:
            rid: request id
            data: data to store
            offset: store offset in bytes
            
        Returns:
            bool: True if success, False otherwise
        """
        if not self.enable_manager or not self.pre_allocate:
            return True
            
        with self._lock:
            if rid not in self._memory_blocks:
                logger.error(f"request {rid} not found in memory blocks")
                return False
                
            memory_block = self._memory_blocks[rid]
            data_size = len(data)
            
            # 检查是否超出预分配的内存大小
            if offset + data_size > len(memory_block):
                logger.error(f"data size {data_size} greater than memory block {len(memory_block)}")
                return False
                
            # 将数据复制到内存块中
            try:
                memory_block[offset:offset+data_size] = np.frombuffer(data, dtype=np.uint8)
                # 记录数据的偏移量和大小
                self._data_offsets[rid] = (offset, data_size)
                logger.debug(f"success {data_size} bytes data store to {rid} memory block at offset {offset}")
                return True
            except Exception as e:
                logger.error(f"store data got error: {e}")
                return False
    
    def retrieve_data(self, rid: str, size: Optional[int] = None, offset: Optional[int] = None) -> Optional[bytes]:
        """retrieve data from memory block
        
        Args:
            rid: request id
            size: retrieve size in bytes, if None, retrieve all stored data
            offset: retrieve offset in bytes, if None, use the offset recorded during store_data
            
        Returns:
            bytes: retrieved data, None if failed
        """
        if not self.enable_manager or not self.pre_allocate:
            return None
            
        with self._lock:
            if rid not in self._memory_blocks:
                logger.error(f"request {rid} not found in memory blocks")
                return None
                
            memory_block = self._memory_blocks[rid]
            
            # 如果没有指定偏移量，使用存储时记录的偏移量
            if offset is None:
                if rid not in self._data_offsets:
                    logger.error(f"no data offset record found for request {rid}")
                    return None
                stored_offset, stored_size = self._data_offsets[rid]
                offset = stored_offset
                # 如果没有指定大小，使用存储时记录的大小
                if size is None:
                    size = stored_size
            else:
                # 如果指定了偏移量但没有指定大小，则检索从偏移量开始的所有数据
                if size is None:
                    size = len(memory_block) - offset
                
            # 检查是否超出预分配的内存大小
            if offset + size > len(memory_block):
                logger.error(f"retrieve range [{offset}:{offset+size}] exceeds memory block size {len(memory_block)}")
                return None
                
            # 从内存块中检索数据
            try:
                data = bytes(memory_block[offset:offset+size])
                logger.debug(f"successfully retrieved {size} bytes data from {rid} memory block at offset {offset}")
                return data
            except Exception as e:
                logger.error(f"error retrieving data: {e}")
                return None

    def refresh_memory_limit(self):
        """refresh memory limit"""
        if not self.enable_manager:
            return
            
        with self._lock:
            old_limit = self.available_memory_bytes
            self._calculate_memory_limit()
            if old_limit != self.available_memory_bytes:
                logger.info(f"memory limit refreshed: {old_limit} -> {self.available_memory_bytes}")

    def start_memory_monitor(self, interval_seconds: int):
        """start memory monitor
        
        Args:
            interval_seconds: refresh interval in seconds
        """
        if not self.enable_manager:
            return
            
        def _monitor_task():
            while True:
                try:
                    self.refresh_memory_limit()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"refresh memory got error: {e}")
                    time.sleep(interval_seconds)
                
        monitor_thread = threading.Thread(target=_monitor_task, daemon=True)
        monitor_thread.start()
        logger.info(f"memory monitor started, refresh interval: {interval_seconds}秒")
        
        self._monitor_thread = monitor_thread
        
    def get_memory_stats(self):
        """get memory stats
        
        Returns:
            dict: memory stats
        """
        if not self.enable_manager:
            return {
                "enabled": False
            }
            
        with self._lock:
            return {
                "enabled": True,
                "total_system_memory": psutil.virtual_memory().total,
                "available_system_memory": psutil.virtual_memory().available,
                "memory_limit": self.available_memory_bytes,
                "total_allocated": self.total_allocated_memory,
                "num_requests": len(self.allocated_memory),
                "remaining_limit": self.available_memory_bytes - self.total_allocated_memory,
                "pre_allocate": self.pre_allocate
            }