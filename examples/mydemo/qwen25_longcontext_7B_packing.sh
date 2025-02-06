export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_11
export NCCL_ALGO=nvls
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_QPS_PER_CONNECTION=2
export WANDB_API_KEY="" # wandb 无效
export WANDB_NAME=""
export WANDB_PROJECT=""
export NCCL_DEBUG=INFO
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1280000

MASTER_ADDR=$(cat /etc/aistudio/master-host)
export ADDR=$MASTER_ADDR
export NODE_RANK=$RANK
export NNODES=$WORLD_SIZE
export NPROC_PER_NODE=8
export NCCL_DEBUG=INFO
export PORT=20023

export TRANSFORMERS_CACHE=/chubao/tj-train-ssd-21/wanghaotian/cache/huggingface/hub/
export HUGGINGFACE_HUB_CACHE=/chubao/tj-train-ssd-21/wanghaotian/cache/huggingface/hub/
export HF_HUB_CACHE=/chubao/tj-train-ssd-21/wanghaotian/cache/huggingface/hub/
export HF_DATASETS_CACHE=/chubao/tj-train-ssd-21/wanghaotian/cache/huggingface/datasets/


xtuner train /chubao/tj-data-ssd-03/liuchengwei/workspaces/xtuner/qwen25_longcontext_7B_packing.py --deepspeed deepspeed_zero3
