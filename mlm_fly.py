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

def torch_mask_tokens(inputs, special_tokens = [101, 102], vocab_size=30000):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    mlm_probability=0.15
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    special_tokens_mask = [
        val in special_tokens for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = 103 #self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, masked_indices*1


class Trainer():
  def __init__(self, tokenizer):
    self.tokenizer=tokenizer
    self.train_data = None
    self.train_dataloader = None
  
  def process_data_to_model_inputs(self, batch):
    tokens = torch.tensor(self.tokenizer.batch_encode_plus(batch["text"], padding="max_length", truncation=True, max_length=128)["input_ids"])
    inputs, labels, mask = torch_mask_tokens(tokens, vocab_size=len(self.tokenizer))
    batch["input"] = inputs
    batch["mask"] = mask
    batch["labels"] = labels
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
        #outputs = outputs.reshape(outputs.size(0)*outputs.size(1), -1)  # (batch * seq_len x classes)
        #label = label.reshape(-1)
        # label = label * mask.reshape(-1)

        # outputs = outputs.detach().cpu()
        loss = criterion(outputs.transpose(1, 2), label)

        # accuracy = jnp.equal(jnp.argmax(logits, axis=-1), label) * label_mask

        loss.backward()
        optimizer.step()
        scheduler.step()
        log_ = {
            'Train Loss': loss,
            'Step': step,
            'LR': scheduler.get_last_lr()[0]
        }
        pbar.set_postfix(log_)


        
