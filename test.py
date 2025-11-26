from loguru import logger
import torch
from dataset import UnifiedSFTDataset
from transformers import AutoModelForCausalLM


import time
from tool import *
from dataset import UnifiedSFTDataset
from collator import SFTDataCollatorDev
from trainer import Escofier_Trainer



def main():
    # 记录开始时间
    first_start = time.time()

    # 获取args, 可配置args_path传入函数
    args, training_args = getArgument()
    
    # 保存文件重命名参数
    model_name = args.model_name_or_path.split('/')[-1]
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    output_dir = training_args.output_dir

    # 获取tokenizer和model
    model = AutoModelForCausalLM.from_pretrained(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = getTokenizer(args.model_name_or_path)


    # 准备数据
    dataset_args = {'args': args, 'tokenizer': tokenizer}
    # Test
    test_dataset = UnifiedSFTDataset(args.test_path, is_train=False, **dataset_args)
    test_dataloader = get_dataloader(training_args, test_dataset, False)
    
    trainer_args = {
        "device": device,
        "scaler": None,
        "optimizer": None,
        "lr_scheduler": None,
        "training_args": None,
    }


    # 准备测试
    logger.info("测试开始")
    trainer = Escofier_Trainer(model, **trainer_args)
    gold_label_list, pred_label_list = trainer.test(test_dataloader, tokenizer.eos_token, tokenizer)
    
    # 保存推理结果
    with open(f"{output_dir}/test_pred{model_name}_rank{lora_rank}_alpha{lora_alpha}.json", 'w', encoding='utf-8') as f:
        json.dump({"gold_label": gold_label_list}, f, indent=4)
    with open(f"{output_dir}/test_gold{model_name}_rank{lora_rank}_alpha{lora_alpha}.json", 'w', encoding='utf-8') as f:
        json.dump({"pred_label": pred_label_list}, f, indent=4)
    end = time.time()
    P, R, F1 = get_prf(gold_label_list, pred_label_list)
    logger.info(f"最终测试集的P: {P:.2f}, P: {R:.2f}, F: {F1:.2f}")
    # swanlab.log({"test_P": P, "test_R": R, "test_F1": F1})


    logger.info(f"本次测试一共花费了{(end - first_start)/60:.2f}分钟")
    




if __name__ == "__main__":
    main()

