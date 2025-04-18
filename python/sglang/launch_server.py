"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree, setup_child_process_monitor

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        setup_child_process_monitor()
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
