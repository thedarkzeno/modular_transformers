import torch
from torch import Tensor
from typing import Any, Dict, Tuple, Union
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import math
from accelerate import Accelerator


class Trainer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_data = None
        self.train_dataloader = None
        self.eval_data = None
        self.eval_dataloader = None
        self.lower = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _tokenize_dataset(self, batch):
        if self.lower:
            tokens = torch.tensor(self.tokenizer.batch_encode_plus(
                [txt.lower() for txt in batch["text"]], padding="max_length", truncation=True, max_length=self.max_length)["input_ids"])
        else:
            tokens = torch.tensor(self.tokenizer.batch_encode_plus(
                batch["text"], padding="max_length", truncation=True, max_length=self.max_length)["input_ids"])
        batch["input"] = tokens.tolist()
        return batch

    def _prepare_dataset(self, file, batch_size, num_workers):
        data = load_dataset(
            'text', data_files={'train': file})

        data = data.map(
            self._tokenize_dataset,
            batched=True,
            num_proc=8,
            batch_size=batch_size,
        )
        data.set_format(type='torch', columns=['input'])

        dataloader = DataLoader(
            data["train"], shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        return data, dataloader

    def prepare_dataset(self, train_file, batch_size, num_workers=0, max_length=128, val_file=None, lower_case=False):
        if lower_case:
            self.lower = True
        self.max_length = max_length

        # prepare the train data
        self.train_data, self.train_dataloader = self._prepare_dataset(train_file, batch_size=batch_size, num_workers=num_workers)

        # prepare validation data
        if val_file is not None:
            self.eval_data, self.eval_dataloader = self._prepare_dataset(val_file, batch_size=batch_size, num_workers=num_workers)

    def eval_step(self, model: nn.Module):
        model.eval()
        losses = []
        for eval_step, eval_batch in enumerate(self.eval_dataloader):
            inputs, labels, mask = torch_mask_tokens(
                        eval_batch['input'], tokenizer=self.tokenizer)
            with torch.no_grad():
                loss = model(inputs, mask=mask, labels=labels)
            losses.append(loss.item())

        #losses = torch.cat(losses)
        losses = torch.tensor(losses)
        losses = losses[: len(self.eval_data)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity


    def fit_mlm(self, model, model_name, epochs=1, accumulation_steps=1, learning_rate=1e-4, warmup_steps=1000, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=0.01, eval_steps=1000, device="cpu", ckpt_steps=0, use_amp=False):
        assert (
            self.train_dataloader is not None), "You need to prepare the dataset first"


        accelerator = Accelerator(fp16=use_amp)
        optimizer = torch.optim.AdamW(model.parameters(
        ), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
        steps = epochs * self.train_dataloader.dataset.num_rows//self.train_dataloader.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, steps)

        #move model, optimizer and data to devices
        model, optimizer, self.train_dataloader = accelerator.prepare(
            model, optimizer, self.train_dataloader)
        # training loop
        with tqdm(total=steps, desc="Steps") as pbar:
            perplexity = 0
            model.train()
            for epoch in range(epochs):
                for step, batch in enumerate(self.train_dataloader):

                    inputs, labels, mask = torch_mask_tokens(
                        batch['input'], tokenizer=self.tokenizer)

                    loss = model(inputs, mask=mask, labels=labels)[0]
                    loss = loss / accumulation_steps

                    accelerator.backward(loss)

                    # Wait for several backward steps
                    if (step+1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                    # eval step
                    if self.eval_dataloader is not None and (step+1) % eval_steps == 0:
                        perplexity = self.eval_step(model)

                    if perplexity != 0:
                        log_ = {
                            'Train Loss': accumulation_steps*loss.item(),
                            'Step': step,
                            'LR': scheduler.get_last_lr()[0],
                            'perplexity': perplexity
                        }
                    else:
                        log_ = {
                            'Train Loss': accumulation_steps*loss.item(),
                            'Step': step,
                            'LR': scheduler.get_last_lr()[0]
                        }
                    if ckpt_steps > 0:
                        if step > 0 and step % ckpt_steps == 0:
                            # save
                            print("saving checkpoint {}".format(step))
                            model.save_pretrained(
                                model_name+"/ckpts/{}".format(step))

                    pbar.set_postfix(log_)
                    pbar.update(1)
