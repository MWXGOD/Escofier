import argparse
from loguru import logger
import os
from os.path import join
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from argument import CustomizedArguments
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
import json
from trl import get_kbit_device_map
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import seed_worker


def find_all_adapter_names(model, train_mode):
    assert train_mode in ['lora', 'qlora']
    target_linear_module = bnb.nn.Linear4bit if train_mode=='qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, target_linear_module):
            sep_name = name.split('.')
            lora_module_names.add(sep_name[0] if len(sep_name)==1 else sep_name[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def getTokenizer(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )
    return tokenizer


def getArgument(train_args_file = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default="./config/qwen2.5-7b-sft-qlora.json")
    args = parser.parse_args()
    train_args_file = train_args_file or args.train_args_file
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)

    # check some setting
    assert args.task_type in ['pretrain', 'sft', 'dpo'], "task_type should be in ['pretrain', 'sft', 'dpo']"
    assert args.train_mode in ['full', 'lora', 'qlora'], "task_type should be in ['full', 'lora', 'qlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, "only one of fp16 and bf16 can be True"

    return args, training_args


def getModel(args, training_args):
    # 断言模型精度
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path.split('/')[-1]}')
    logger.info(f'Train mode with {args.train_mode}')

    # 获取模型精度
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16

    # 设定模型模式
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
        ) if args.train_mode == 'qlora' else None
    
    model_kwargs = dict(
        trust_remote_code=True,
        # attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.train_mode == 'qlora' and  args.task_type in ['sft', 'pretrain']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 找到所有需要adapter的Linear,构建peft模型
    if args.train_mode == 'full':
        peft_config = None
    else:
        target_model = find_all_adapter_names(model, args.train_mode)
        peft_config = LoraConfig(
            r = args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_model,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
        )
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrian', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # init ref_model
    if args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs) if args.train_mode == 'full' else None
    # pretrain和sft，不需要ref_model
    else:
        ref_model = None

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config
    }


def init_lora_components(args, training_args):
    # 初始化组件
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    # tokenizer = getTokenizer(args.model_name_or_path)

    # 加载dataset和collator
    # if args.task_type == 'sft':
    #     kwargs = {
    #         'tokenizer': tokenizer
    #     }
    #     logger.info('Train model with sft task')
    #     logger.info('Loading data with UnifiedSFTDataset')
    #     train_dataset = UnifiedSFTDataset(args, **kwargs)
    #     logger.info('Loading collator with SFTDataCollator')
    #     data_collator = SFTDataCollator(args, **kwargs)
    # else:
    #     logger.info('仅支持sft微调')

    # 加载model
    model_dict = getModel(args, training_args)
    model, ref_model, peft_config = model_dict['model'], model_dict['ref_model'], model_dict['peft_config']


    return model, ref_model, peft_config


def get_dataloader(args, dataset, data_collator, is_training):
    
    if not isinstance(dataset, IterableDataset):
        sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=args.per_device_train_batch_size, 
            sampler=sampler,
            drop_last=is_training,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker if is_training else None,
            collate_fn=data_collator
        )
    else:
        dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size)

    return dataloader


def get_prf(gold_label_list, pred_label_list):
    P, R, C, F = 0, 0, 0, 0

    for gold_label, pred_label in zip(gold_label_list, pred_label_list):
        if 'none' not in str(gold_label).lower():
            gold_label = [gl.strip() for gl in gold_label]
            pred_label = [pl.strip() for pl in pred_label]
            gold_label = set(gold_label)
            pred_label = set(pred_label)
            P += len(pred_label)
            R += len(gold_label)
            C += len(pred_label & gold_label)

    P = C / P if P > 0 else 0
    R = C / R if R > 0 else 0
    F = 2 * P * R / (P + R) if P + R > 0 else 0
    return P, R, F
        
    


