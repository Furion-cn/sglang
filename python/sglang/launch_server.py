"""Launch the inference server."""

import os
import sys
import threading

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree, monitor_children_and_exit_on_failure

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    parent_pid = os.getpid()
    threading.Thread(target=monitor_children_and_exit_on_failure(parent_pid), daemon=True).start()
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(parent_pid, include_parent=False)
