import json,os
from typing import Any, Dict, List
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class UnifiedSFTDataset(Dataset):
    def __init__(self, data_path, is_train, **kwargs):
        # 配置组件
        self.data_path = data_path
        self.is_train = is_train

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.tokenizer = kwargs.get("tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = self.getTokenizer()
        
        self.data_list = self.getData()
        logger.info('Loading Dataset tokenizer: {}'.format(self.args.model_name_or_path))
        logger.info('Loading Dataset data: {}'.format(self.data_path))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        input_ids, target_mask = [], []

        # instruction = data['instruction']
        input = data['instruction'] + data['input']
        output = data['output'] + self.tokenizer.eos_token
        # qwen里面没有bos_token
        input_tokens = self.tokenizer.encode(input, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(output, add_special_tokens=False)
        input_ids = input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
        assert len(input_ids) == len(target_mask)
        
        # 对长度进行截断
        input_ids = input_ids[:self.args.max_seq_length]
        target_mask = target_mask[:self.args.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs

    def getTokenizer(self):
        # 加载tokenzier
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True
        )
        return tokenizer
    
    def getData(self):
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"❌ 未找到数据文件: {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        return data_list
    
    def collate_fn(self, batch: List[Dict[str, Any]]):
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        batch_max_len = min(max(lengths), self.args.max_seq_length)

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

            if not self.is_train:
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
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.args.max_seq_length]
            attention_mask = attention_mask[:self.args.max_seq_length]
            target_mask = target_mask[:self.args.max_seq_length]
            # 加入list
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)
        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        labels_batch = torch.where(target_mask_batch == 1, input_ids_batch, -100)

        if not self.is_train:
            # generate做padding
            gen_lengths = [len(x) for x in generate_input_ids_list]
            gen_max_len = min(max(gen_lengths), self.args.max_seq_length)

            gen_input_ids_batch, gen_attention_mask_batch = [], []
            for g_ids, g_mask in zip(generate_input_ids_list, generate_attention_mask_list):
                pad_len = gen_max_len - len(g_ids)
                g_ids = [self.tokenizer.pad_token_id] * pad_len + g_ids
                g_mask = [0] * pad_len + g_mask

                gen_input_ids_batch.append(g_ids)
                gen_attention_mask_batch.append(g_mask)
            
            gen_input_ids_batch = torch.tensor(gen_input_ids_batch, dtype=torch.long)
            gen_attention_mask_batch = torch.tensor(gen_attention_mask_batch, dtype=torch.long)

            inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
                "target_mask": target_mask_batch,
                "generate_input_ids": gen_input_ids_batch,     # 给 model.generate 用
                "generate_attention_mask": gen_attention_mask_batch,
                "output_tokens": output_tokens_list,           # answer 的 token id 序列
            }
        else:
            inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch
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
    # print((train_dataset[0]))
    pass






