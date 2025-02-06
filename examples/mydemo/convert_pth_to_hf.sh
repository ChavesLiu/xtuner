
CONFIG_PATH="/chubao/tj-train-ssd-21/wanghaotian/scripts/xtuner/boot_train/qwen2_longcontext_72B_10w_packing_v8-like.py"
CHECKPOINT_PATH="/chubao/tj-train-ssd-21/wanghaotian/scripts/xtuner/boot_train/work_dirs/qwen2_longcontext_72B_10w_packing_v8-like/iter_371.pth"
OUTPUT_PATH="/chubao/tj-train-ssd-21/wanghaotian/scripts/xtuner/boot_train/hf_models/qwen25-72B-like-v8_371"

MAX_JOBS=64 xtuner convert pth_to_hf $CONFIG_PATH $CHECKPOINT_PATH $OUTPUT_PATH
