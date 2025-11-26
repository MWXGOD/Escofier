from typing import Any, Dict, List
import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

# 已弃用！！！！！！！！！！！！！！！！！见dataset-conllate_fn

class SFTDataCollatorTrain(object):
    def __init__(self, config, **kwargs):
        # 配置组件
        if kwargs['tokenizer']:
            self.tokenizer = kwargs['tokenizer']
        else:
            self.tokenizer = self.getTokenizer(config.model_name_or_path)
        # logger.info('Loading Collator data: {}'.format(config.data_path))
    
        self.max_seq_length = config.max_seq_length
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)

            # padding, 如果为负数，会返回空列表，即不进行填充
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            # input_ids = input_ids[:self.max_seq_length]
            # attention_mask = attention_mask[:self.max_seq_length]
            # target_mask = target_mask[:self.max_seq_length]
            # 加入list
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs

    def getTokenizer(self, model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        # 加载tokenzier
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            # llama不支持fast
            use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
        )
        return tokenizer

class SFTDataCollatorDev(SFTDataCollatorTrain):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        generate_input_ids_list = []
        generate_attention_mask_list = []
        output_tokens_list = []        

        # 输入padding，并且提取生成的list
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)

            # 找到prompt和answer
            # prompt
            first_answer_index = None
            for i, v in enumerate(target_mask):
                if v == 1:
                    first_answer_index = i
                    break
            if first_answer_index == None:
                prompt_ids = input_ids
                prompt_attn = attention_mask
                answer_ids = []
            else:
                prompt_ids = input_ids[:first_answer_index]
                prompt_attn = attention_mask[:first_answer_index]
                answer_ids = input_ids[first_answer_index:]
            generate_input_ids_list.append(prompt_ids)
            generate_attention_mask_list.append(prompt_attn)
            output_tokens_list.append(answer_ids)

            # padding, 如果为负数，会返回空列表，即不进行填充
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            # input_ids = input_ids[:self.max_seq_length]
            # attention_mask = attention_mask[:self.max_seq_length]
            # target_mask = target_mask[:self.max_seq_length]
            # 加入list
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)
        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)

        # generate做padding
        gen_lengths = [len(x) for x in generate_input_ids_list]
        gen_max_len = min(max(gen_lengths), self.max_seq_length)

        gen_input_ids_batch, gen_attention_mask_batch = [], []
        for g_ids, g_mask in zip(generate_input_ids_list, generate_attention_mask_list):
            pad_len = gen_max_len - len(g_ids)
            g_ids = [self.pad_token_id] * pad_len + g_ids
            g_mask = [0] * pad_len + g_mask

            gen_input_ids_batch.append(g_ids)
            gen_attention_mask_batch.append(g_mask)
        
        gen_input_ids_batch = torch.tensor(gen_input_ids_batch, dtype=torch.long)
        gen_attention_mask_batch = torch.tensor(gen_attention_mask_batch, dtype=torch.long)

        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
            "target_mask": target_mask_batch,
            "generate_input_ids": gen_input_ids_batch,     # 给 model.generate 用
            "generate_attention_mask": gen_attention_mask_batch,
            "output_tokens": output_tokens_list,           # answer 的 token id 序列
        }
        return inputs















if __name__ == "__main__":
    # with open("/home/muwenxuan/Escofier/data/sample_data/dummy_data.jsonl", 'r', encoding='utf-8') as f:
    #     data = f.readlines()
    #     data = [json.loads(d) for d in data]
    # # print(type(data))
    # with open("/home/muwenxuan/Escofier/data/sample_data/dummy_data.jsonl", 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)
    
    # config_path = "/home/muwenxuan/Escofier/config/qwen2.5-0.5b-sft-qlora.json"
    # config = Config(config_path)
    # train_dataset = UnifiedSFTDataset(config)
    # collotor = SFTDataCollator(config)
    # out = collotor(train_dataset)

    # print(out[0])
    pass
