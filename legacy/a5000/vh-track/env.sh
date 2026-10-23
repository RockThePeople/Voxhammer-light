# VoxHammer 전용 환경 — CUDA 11.8(torch cu118 매칭). 시스템 전역 CUDA 12.9는 건드리지 않음.
export CUDA_HOME=$HOME/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=<HOME>/ai-ar-v2/eigen:$CPLUS_INCLUDE_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
