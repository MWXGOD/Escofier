from tool import *
from torch.amp import autocast
import swanlab
from tqdm.auto import tqdm

class Escofier_Trainer:
    def __init__(self, model=None, **kwargs):
        self.model = model

        # 循环设置 kwargs 中的所有键值对为成员变量
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def train(self, epoch, train_dataloader):
        self.model.train()
        loss_per_epoch = 0.0

        train_dataloader_with_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{self.training_args.num_train_epochs}",
            unit="batch"
        )

        for step, batch in train_dataloader_with_bar:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(**batch)
                train_loss = outputs.loss
                train_loss_accumulate = train_loss / self.training_args.gradient_accumulation_steps
            self.scaler.scale(train_loss_accumulate).backward()

            if (step +1) % self.training_args.gradient_accumulation_steps == 0 or (step +1) == len(train_dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            loss_per_epoch += train_loss.item()
            swanlab.log({"train_loss_per_step": train_loss.item()})
            
        avg_loss_per_epoch = loss_per_epoch / len(train_dataloader)
        return avg_loss_per_epoch
            
    def dev(self, epoch, dev_dataloader, stop_word, tokenizer):
        self.model.eval()
        top_p = 0.9  
        temperature = 0.7  
        epoch_loss_dev = 0
        max_new_tokens = 1024
        gold_label_list = []
        pred_label_list = []
        stop_token_id = tokenizer.convert_tokens_to_ids(stop_word)
        with torch.no_grad():
            for index, batch in tqdm(
                enumerate(dev_dataloader), 
                total=len(dev_dataloader),
                desc=f"Epoch {epoch+1}/{self.training_args.num_train_epochs}",
                unit="batch"
            ):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(**batch)
                    dev_loss = outputs.loss
                epoch_loss_dev += dev_loss.item()
                swanlab.log({"dev_loss_per_step": dev_loss.item()})

                generate_input_ids = batch['generate_input_ids']
                generate_attention_mask = batch['generate_attention_mask']
                outputs = self.model.generate(  # 能够实现批量输出
                    input_ids=generate_input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                    top_p=top_p, temperature=temperature, eos_token_id=stop_token_id, pad_token_id=tokenizer.pad_token_id, attention_mask=generate_attention_mask
                )
                gold_labels = batch['output_tokens']
                gold_labels = tokenizer.batch_decode(gold_labels,skip_special_tokens=True)
                pred_label_logits = outputs[:, len(generate_input_ids[0]):]
                predicted_labels = tokenizer.batch_decode(pred_label_logits,skip_special_tokens=True)
                for gl, pl in zip(gold_labels, predicted_labels):
                    gold_label_list.append(gl.split('##'))
                    pred_label_list.append(pl.split('##'))
        avg_dev_epoch_loss = epoch_loss_dev / len(dev_dataloader)
        return avg_dev_epoch_loss, gold_label_list, pred_label_list
            
    def test(self, test_dataloader, stop_word, tokenizer):
        self.model.eval()
        top_p = 0.9  
        temperature = 0.7  
        max_new_tokens = 1024
        gold_label_list = []
        pred_label_list = []
        stop_token_id = tokenizer.convert_tokens_to_ids(stop_word)
        with torch.no_grad():
            for index, batch in tqdm(
                enumerate(test_dataloader), 
                total=len(test_dataloader),
                desc=f"Testing...",
                unit="batch"
            ):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                generate_input_ids = batch['generate_input_ids']
                generate_attention_mask = batch['generate_attention_mask']
                outputs = self.model.generate(  # 能够实现批量输出
                    input_ids=generate_input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                    top_p=top_p, temperature=temperature, eos_token_id=stop_token_id, pad_token_id=tokenizer.pad_token_id, attention_mask=generate_attention_mask
                )
                gold_labels = batch['output_tokens']
                gold_labels = tokenizer.batch_decode(gold_labels,skip_special_tokens=True)
                pred_label_logits = outputs[:, len(generate_input_ids[0]):]
                predicted_labels = tokenizer.batch_decode(pred_label_logits,skip_special_tokens=True)
                for gl, pl in zip(gold_labels, predicted_labels):
                    gold_label_list.append(gl.split('##'))
                    pred_label_list.append(pl.split('##'))
        return gold_label_list, pred_label_list
