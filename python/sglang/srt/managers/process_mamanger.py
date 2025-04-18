import logging
import os
import signal
import sys

logger = logging.getLogger(__name__)

def setup_child_process_monitor():
    def child_handler(signum, frame):
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid > 0:
                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                signal_num = os.WTERMSIG(status) if os.WIFSIGNALED(status) else -1

                if exit_code != 0 or signal_num != -1:
                    logger.error(f"Child process {pid} terminated with exit code {exit_code} and signal {signal_num}")

                    if exit_code == 131 or signal_num in (4, 6, 11):
                        logger.critical("Critical error detected, main process will exit...")
                        os.kill(os.get_pid(), signal.SIGTERM)
                        sys.exit(1)
        except Exception as e:
            logger.error(f"Error in child process monitor: {e}")
    
    signal.signal(signal.SIGCHLD, child_handler)