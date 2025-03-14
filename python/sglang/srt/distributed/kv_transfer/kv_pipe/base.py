# SPDX-License-Identifier: Apache-2.0
"""
This file defines an interface `KVPipeBase`
that provides an abstraction for sending and receiving tensors, or None, via
distributed communications.

All classes instantiated from this interface are assumed to be a FIFO pipe.

If your distributed communication platform already supports key-value lookup,
you can bypass this interface and directly start from `kv_lookup_buffer`.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class KVTransferAgent(ABC):
    """
    This class provides an interface for sending and receiving tensors, or
    None, by distributed communications.
    """

    @abstractmethod
    def send_kv_cache(self, dst_prt: int, tensor: Optional[torch.Tensor]) -> None:
        """Send a tensor, or None, via the pipe.
        
        Need to support sending None -- important for error handling.
        
        TODO: add a `key` argument so that we can use traditional 
        key-value database as the distributed communication mechanism behind 
        the pipe.

        Args:
            dst_prt (int): The destination buffer pointer address.
            tensor (Optional[torch.Tensor]): The tensor to be sent. Can be None.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_kv_cache(self, socket_addr: str, rid: str, length: int) -> Optional[torch.Tensor]:
        """Receive a tensor (can be None) from the pipeline.
        Args:
            socket_addr (str): The control-plan socket address of kv cache sourcing node.
            rid (str): The request id of the kv cache.
            length (int): The length of the tensor to be received.
        Returns:
            Optional[torch.Tensor]: The tensor received from the pipeline. Can 
                                    be None.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the pipeline and release resources.

        This method is responsible for closing the communication pipeline 
        and releasing any resources associated with it.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError
