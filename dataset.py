import json,os
from loguru import logger
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

class UnifiedSFTDataset(Dataset):
    def __init__(self, config, data_path, **kwargs):
        # 配置组件
        if kwargs['tokenizer']:
            self.tokenizer = kwargs['tokenizer']
        else:
            self.tokenizer = self.getTokenizer(config.model_name_or_path)
        self.data_list = self.getData(data_path)
        logger.info('Loading Dataset tokenizer: {}'.format(config.model_name_or_path))
        logger.info('Loading Dataset template: {}'.format(config.template_name))
        logger.info('Loading Dataset data: {}'.format(data_path))

        self.max_seq_length = config.max_seq_length

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
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
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
    
    def getData(self, data_path):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"❌ 未找到数据文件: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        return data_list




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






