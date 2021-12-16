import torch
from torch import Tensor
from typing import Tuple
from datasets import load_dataset, load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import math


def getIgnoreMask(tokens_tensor, ignore_tokens):
    result = []
    finished = False
    for v in tokens_tensor:
        if finished == False:
            result.append(v in ignore_tokens)
            if v == 0:  # Once it gets to the padding you don't need to check anymore
                finished = True
        else:
            result.append(True)
    return result


def torch_mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    # self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    inputs[indices_replaced] = 103

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, masked_indices*1


class Trainer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_data = None
        self.train_dataloader = None
        self.eval_data = None
        self.eval_dataloader = None
        self.lower = False

    def process_data_to_model_inputs(self, batch):
        if self.lower:
            tokens = torch.tensor(self.tokenizer.batch_encode_plus(
                [txt.lower() for txt in batch["text"]], padding="max_length", truncation=True, max_length=self.max_length)["input_ids"])
        else:
            tokens = torch.tensor(self.tokenizer.batch_encode_plus(
                batch["text"], padding="max_length", truncation=True, max_length=self.max_length)["input_ids"])
        inputs, labels, mask = torch_mask_tokens(
            tokens, tokenizer=self.tokenizer)
        batch["input"] = inputs
        batch["mask"] = mask
        batch["labels"] = labels
        return batch

    def prepare_dataset(self, train_file, batch_size, max_length=128, val_file=None, lower_case=False):
        self.max_length = max_length
        self.train_data = load_dataset(
            'text', data_files={'train': train_file})
        if lower_case:
            self.lower = True
        self.train_data = self.train_data.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
        )
        self.train_data.set_format(type='torch', columns=[
                                   'input', 'labels', 'mask'])
        self.train_dataloader = DataLoader(
            self.train_data["train"], shuffle=True, batch_size=batch_size
        )
        if val_file is not None:
            self.eval_data = load_dataset("text", data_files={'val': val_file})
            self.eval_data = self.eval_data.map(
                self.process_data_to_model_inputs,
                batched=True,
                batch_size=batch_size,
            )
            self.eval_data.set_format(type='torch', columns=[
                                       'input', 'labels', 'mask'])
            self.eval_dataloader = DataLoader(
                self.eval_data["val"], shuffle=True, batch_size=batch_size
            )

    def fit_mlm(self, model, model_name, epochs=1, learning_rate=1e-4, warmup_steps=1000, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=0.01, eval_steps=1000, device="cpu", ckpt_steps=0, use_amp=False):
        assert (
            self.train_dataloader is not None), "You need to prepare the dataset first"
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.AdamW(model.parameters(
        ), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
        steps = epochs * self.train_dataloader.dataset.num_rows//self.train_dataloader.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, steps)
        model.to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # training loop
        # pbar = tqdm(range(epochs), desc="Steps")
        with tqdm(total=steps, desc="Steps") as pbar:
            perplexity = 0
            for epoch in range(epochs):
                for step, batch in enumerate(self.train_dataloader):
                    model.train()
                    # optimizer.zero_grad()
                    # batch.set_format(type='torch', columns=['input', 'labels', 'mask'])
                    # model(batch['input'].to(device), batch['mask'].to(device))
                    label = batch['labels'].to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(batch['input'].to(device))
                        loss = criterion(outputs.transpose(1, 2), label)

                    # accuracy = jnp.equal(jnp.argmax(logits, axis=-1), label) * label_mask
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    # loss.backward()
                    # optimizer.step()

                    # eval step

                    if self.eval_dataloader is not None and (step+1) % eval_steps == 0:
                        print("evaluating...")
                        model.eval()
                        losses = []
                        for eval_step, eval_batch in enumerate(self.eval_dataloader):
                            with torch.no_grad():
                                outputs = model(eval_batch['input'].to(device))
                            label = batch['labels'].to(device)
                            loss = criterion(outputs.transpose(1, 2), label)
                            losses.append(loss)

                        #losses = torch.cat(losses)
                        losses = torch.tensor(losses)
                        losses = losses[: len(self.eval_data)]
                        try:
                            perplexity = math.exp(torch.mean(losses))
                        except OverflowError:
                            perplexity = float("inf")
                    if perplexity != 0:
                        log_ = {
                            'Train Loss': loss,
                            'Step': step,
                            'LR': scheduler.get_last_lr()[0],
                            'perplexity': perplexity
                        }
                    else:
                        log_ = {
                            'Train Loss': loss,
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
