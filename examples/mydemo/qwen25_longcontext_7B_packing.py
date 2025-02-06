# Copyright (c) OpenMMLab. All rights reserved.
from datasets import load_dataset
from mmengine.config import ConfigDict
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
def sharegpt_map_fn_single_turn(example):
    convs = example["conversations"]
    q,a = convs[0]["value"], convs[1]["value"]
    system = example["system"]
    return {
        'conversation': [{
            'system': system,
            'input': q,
            'output': a
        }]
    }


# 加入了observation的处理逻辑，可能会影响summary效果 暂时看来影响应该有限
def sharegpt_map_fn_multi_turn(example):
    messages = example['conversations']
    system = ''
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'assistant':
        # Skip the first one if it is from assistant
        messages = messages[1:]
    for i,msg in enumerate(messages):
        if msg['from'] == 'human' or msg["from"] == "observation":
            input += msg['value']
        elif msg['from'] == 'assistant':
            if i == 1:
                system = example["system"]
            conversation.append({
                'system': system,
                'input': input,
                'output': msg['value']
            })
            system = ''
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


SYSTEM_TEMPLATE_CUSTOM = ConfigDict(
    moss_sft=('You are an AI assistant whose name is {bot_name}.\n'
              'Capabilities and tools that {bot_name} can possess.\n'
              '- Inner thoughts: enabled.\n'
              '- Web search: enabled. API: Search(query)\n'
              '- Calculator: enabled. API: Calculate(expression)\n'
              '- Equation solver: enabled. API: Solve(equation)\n'
              '- Text-to-image: disabled.\n'
              '- Image edition: disabled.\n'
              '- Text-to-speech: disabled.\n'),
    alpaca=('Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n'),
    arxiv_gentile=('If you are an expert in writing papers, please generate '
                   "a good paper title for this paper based on other authors' "
                   'descriptions of their abstracts.\n'),
    colorist=('You are a professional color designer. Please provide the '
              'corresponding colors based on the description of Human.\n'),
    coder=('You are a professional programer. Please provide the '
           'corresponding code based on the description of Human.\n'),
    lawyer='你现在是一名专业的中国律师，请根据用户的问题给出准确、有理有据的回复。\n',
    medical='如果你是一名医生，请根据患者的描述回答医学问题。\n',
    sql=('If you are an expert in SQL, please generate a good SQL Query '
         'for Question based on the CREATE TABLE statement.\n'),
    default=('You are a helpful assistant.')
)

# Model
# pretrained_model_name_or_path = "/chubao/tj-train-ssd-21/wanghaotian/scripts/LLaMA-Factory/saves/qwen25-72B/Qwen25-72B-dashu-v11-like-v8-tuluif-keqa-func/checkpoint-4988"    # TODO CUSTOM MODEL PATH
pretrained_model_name_or_path = "/chubao/tj-train-ssd-21/wanghaotian/scripts/xtuner/boot_train/hf_models/qwen25-7B-like-v8_744"    # TODO CUSTOM MODEL PATH
use_varlen_attn = False

# Data
path = "/chubao/tj-train-ssd-21/liuchengwei/datasets/memory/memory_core/raw_data/memory-v1.7/train"
# data_path = ['/chubao/tj-train-ssd-21/wanghaotian/scripts/LLaMA-Factory/data/LONG_CONTEXT/long_mix_1108.jsonl']           # TODO CUSTOM DATA PATH
# data_path = ["/chubao/tj-train-ssd-21/wanghaotian/scripts/LLaMA-Factory/data/LONG_CONTEXT/train_data/longcontext_sft_11w_longalign-kelong-general.jsonl"]
filenames = [
        "General_1.jsonl",
        "IF_1.jsonl",
        "IF_2.jsonl",
        "LiveIn.jsonl",
    ]


data_path = [f"{path}/{file}" for file in filenames]

prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 65536 
pack_to_max_length = True

# parallel
sequence_parallel_size = 8

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 1
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 64
max_epochs = 2
optim_type = AdamW
lr = 2e-6
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 0
SYSTEM = SYSTEM_TEMPLATE_CUSTOM.default      # TODO ADD CUSTOM SYSTEM TEMPLATE
evaluation_inputs = [
    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=data_path),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=sharegpt_map_fn_single_turn,    ## 修改数据加载模式
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
