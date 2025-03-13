import torch

class KVTransferAgent:
    def __init__(self):
        self.kv_buffer = []

    def send_kv_cache(self, rid: str,  dst_ptr: int):
        pass

    def recv_kv_cache(self, addr: str, rid: str, length: int) -> torch.Tensor:
        pass