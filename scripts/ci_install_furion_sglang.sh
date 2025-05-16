#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# # Update pip
# pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# # Clean up existing installations
# pip uninstall -y sglang
# #pip cache purge
# # rm -rf /root/.cache/flashinfer
# # rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
# # rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*

# # Install the main package
# HTTP_PROXY='' HTTPS_PROXY='' pip install -e "python[all]" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
