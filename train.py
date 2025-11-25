from loguru import logger
import os
import torch
import bitsandbytes as bnb
from collator import SFTDataCollatorTrain
from dataset import UnifiedSFTDataset
from transformers import (
    AutoModelForCausalLM,
    get_scheduler
)

import time
from tool import *
from dataset import UnifiedSFTDataset
from collator import SFTDataCollatorTrain, SFTDataCollatorDev
from torch.amp import GradScaler
import math
import swanlab
from trainer import Escofier_Trainer
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    # 记录开始时间
    first_start = time.time()

    # 获取args, 可配置args_path传入函数
    args, training_args = getArgument()

    # 创建一个SwanLab项目, 按照时间起名
    time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    swanlab.init(
        project="MWX-Escofier",
        config=args,
        experiment_name=f"MWX-Escofier-{time_str}"
    )

    # 获取tokenizer和model
    model, _, _ = init_lora_components(args, training_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    tokenizer = getTokenizer(args.model_name_or_path)

    # 准备数据
    # Train
    train_dataset_args = {'tokenizer': tokenizer}
    train_dataset = UnifiedSFTDataset(args, args.train_path, **train_dataset_args)
    train_collator_args = {'tokenizer': tokenizer}
    train_collator = SFTDataCollatorTrain(args, **train_collator_args)
    train_dataloader = get_dataloader(training_args, train_dataset, train_collator, True)
    # Dev
    dev_dataset_args = {'tokenizer': tokenizer}
    dev_dataset = UnifiedSFTDataset(args, args.dev_path, **dev_dataset_args)
    dev_collator_args = {'tokenizer': tokenizer}
    dev_collator = SFTDataCollatorDev(args, **dev_collator_args)
    dev_dataloader = get_dataloader(training_args, dev_dataset, dev_collator, False)
    # Test
    test_dataset_args = {'tokenizer': tokenizer}
    test_dataset = UnifiedSFTDataset(args, args.test_path, **test_dataset_args)
    test_collator_args = {'tokenizer': tokenizer}
    test_collator = SFTDataCollatorDev(args, **test_collator_args)
    test_dataloader = get_dataloader(training_args, test_dataset, test_collator, False)


    # Optimizer和scaler
    optimizer = bnb.optim.AdamW8bit(list(model.parameters()), lr=training_args.learning_rate,weight_decay=training_args.weight_decay)
    scaler = GradScaler()

    # lr_scheduler
    if len(train_dataloader) is not None:
        num_optimizer_steps_per_epoch = math.ceil(max(len(train_dataloader) / training_args.gradient_accumulation_steps, 1))
        num_training_steps = math.ceil(num_optimizer_steps_per_epoch * training_args.num_train_epochs)
    else:
        logger.info("------------------------------ train_dataloader is None ------------------------------")
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(num_training_steps * training_args.warmup_ratio),
        num_training_steps=num_training_steps
    )

    # gradient_checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    trainer_args = {
        "device": device,
        "scaler": scaler,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "training_args": training_args,
    }
    trainer = Escofier_Trainer(model, **trainer_args)
    
    # 保存文件重命名参数
    model_name = args.model_name_or_path
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    output_dir = training_args.output_dir

    for epoch in  range(training_args.num_train_epochs): 
        # 准备训练
        logger.info("训练开始")
        start = time.time()
        avg_loss_per_epoch = trainer.train(epoch, train_dataloader)
        end = time.time()
        logger.info(f"当前轮次: {epoch + 1}, 花费了{(end - start)/60:.2f}分钟, 平均训练损失: {avg_loss_per_epoch:.4f}")
        swanlab.log({"train_loss_per_epoch": avg_loss_per_epoch})

        # 准备验证
        logger.info("验证开始")
        start = time.time()
        avg_dev_epoch_loss, gold_label_list, pred_label_list = trainer.dev(epoch, dev_dataloader, tokenizer.eos_token, tokenizer)
        end = time.time()
        P, R, F1 = get_prf(gold_label_list, pred_label_list)
        max_f1 = -1
        if F1 > max_f1:
            max_f1 = F1
            model.save_pretrained(output_dir) 
            with open(f"{output_dir}/dev_gold_{model_name}_rank{lora_rank}_alpha{lora_alpha}.json", 'w', encoding='utf-8') as f:
                json.dump({"gold_label": gold_label_list}, f, indent=4)
            with open(f"{output_dir}/dev_pred_{model_name}_rank{lora_rank}_alpha{lora_alpha}.json", 'w', encoding='utf-8') as f:
                json.dump({"pred_label": pred_label_list}, f, indent=4)
            logger.info("模型已保存")
        end = time.time()
        logger.info(f"当前轮次: {epoch + 1}, 花费了{(end - start)/60:.2f}分钟")
        logger.info(f"平均训练损失: {avg_dev_epoch_loss:.4f}")
        logger.info(f"评价指标P: {P:.2f}, P: {R:.2f}, F: {F1:.2f}")
        swanlab.log({"dev_P": P, "dev_R": R, "dev_F1": F1})

    # 准备测试
    logger.info("测试开始")
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    start = time.time()
    gold_label_list, pred_label_list = trainer.test(epoch, test_dataloader, train_dataset.stop_word, tokenizer)
    
    # 保存推理结果
    with open(f"{output_dir}/test_pred{model_name}_rank{lora_rank}_alpha{lora_alpha}.json", 'w', encoding='utf-8') as f:
        json.dump({"gold_label": gold_label_list}, f, indent=4)
    with open(f"{output_dir}/test_gold{model_name}_rank{lora_rank}_alpha{lora_alpha}.json", 'w', encoding='utf-8') as f:
        json.dump({"pred_label": pred_label_list}, f, indent=4)
    end = time.time()
    P, R, F1 = get_prf(gold_label_list, pred_label_list)
    logger.info(f"最终测试集的P: {P:.2f}, P: {R:.2f}, F: {F1:.2f}")
    swanlab.log({"test_P": P, "test_R": R, "test_F1": F1})


    logger.info(f"本次训练一共花费了{(end - first_start)/60:.2f}分钟")

                    
    
if __name__ == "__main__":
    main()













