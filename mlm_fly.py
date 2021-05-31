import torch
from torch import Tensor
from typing import Tuple
from datasets import load_dataset, load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn

def getIgnoreMask(tokens_tensor, ignore_tokens):
  result = []
  finished = False
  for v in tokens_tensor:
    if finished == False:
      result.append(v in ignore_tokens)
      if v == 0:         ### Once it gets to the padding you don't need to check anymore
        finished = True
    else:
      result.append(True)
  return result

def mask_with_prob(t: Tensor, prob: float, replace_prob: float, ignore_tokens: list = [101, 102, 0], mask_index: int=103) -> Tuple[Tensor, Tensor]:
    probs=torch.zeros_like(t).float().uniform_(0, 1)
    probs2=torch.zeros_like(t).float().uniform_(0, 1)
    non_masked_tokens = probs < 1 - prob
    replace_mask = probs2 < replace_prob

    tokens_tensor = t.detach().clone()
    shape = tokens_tensor.shape
    tokens_tensor = tokens_tensor.reshape(-1)
    ignore_mask = torch.tensor(getIgnoreMask(tokens_tensor, ignore_tokens)).reshape(shape)
    # ignore_mask = torch.tensor([v in ignore_tokens for v in t])

    non_masked_tokens = torch.logical_or(non_masked_tokens, ignore_mask)

    masked_tokens = (~non_masked_tokens)
    non_masked_tokens = torch.logical_and(non_masked_tokens, ~replace_mask)
    mask = torch.tensor([mask_index]) * torch.logical_and(masked_tokens, replace_mask)

    return t*non_masked_tokens + mask, masked_tokens * 1


class Trainer():
  def __init__(self, tokenizer):
    self.tokenizer=tokenizer
    self.train_data = None
    self.train_dataloader = None
  
  def process_data_to_model_inputs(self, batch):
    tokens = torch.tensor(self.tokenizer.batch_encode_plus(batch["text"], padding="max_length", truncation=True, max_length=128)["input_ids"])
    masked, mask = mask_with_prob(tokens, 0.3, 0.5)
    batch["input"] = masked
    batch["mask"] = mask
    batch["labels"] = tokens
    return batch

  def prepare_dataset(self, file, batch_size):
    self.train_data = load_dataset('text', data_files={'train': file})
    self.train_dataloader = DataLoader(
        self.train_data["train"], shuffle=True, batch_size=batch_size
    )
    
  def fit_mlm(self, model, epochs=1, learning_rate=1e-4, warmup_steps=1000, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, device="cpu"):
    assert (self.train_dataloader is not None), "You need to prepare the dataset first"
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
    steps = self.train_dataloader.dataset.num_rows//self.train_dataloader.batch_size 
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)
    model.to(device)

    #training loop
    pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in pbar:
      model.train()
      for step, batch in enumerate(self.train_dataloader):
        optimizer.zero_grad()
        batch = self.process_data_to_model_inputs(batch)
        # batch.set_format(type='torch', columns=['input', 'labels', 'mask'])
        outputs = model(batch['input'].to(device))#model(batch['input'].to(device), batch['mask'].to(device))
        label = batch['labels'].to(device)
        # mask = batch['mask'].to(device)
        outputs = outputs.reshape(outputs.size(0)*outputs.size(1), -1)  # (batch * seq_len x classes)
        label = label.reshape(-1)
        # label = label * mask.reshape(-1)

        # outputs = outputs.detach().cpu()
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()
        scheduler.step()
        log_ = {
            'Train Loss': loss,
            'Step': step,
            'LR': scheduler.get_last_lr()[0]
        }
        pbar.set_postfix(log_)


        
