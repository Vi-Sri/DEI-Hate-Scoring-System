import os

from typing import Tuple, List
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm import tqdm 

input_file_path = './dataset/'
train_df = pd.read_csv(os.path.join(input_file_path, 'train.csv'))
print(train_df.shape)
print(list(train_df.columns))

train_split_df, val_split_df = train_test_split(train_df, test_size=0.05)

print(train_split_df.shape)
print(val_split_df.shape)

bert_model_name = 'bert-base-cased' 
device = torch.device('cuda:0')
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

class DEIHateClassifier(nn.Module):

  def __init__(self, bert: BertModel, num_classes: int):
    super().__init__()
    self.bert = bert
    self.classifier = nn.Linear(bert.config.hidden_size, num_classes)  # in_features, out_features

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
              position_ids=None, head_mask=None, labels=None):
    outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        head_mask=head_mask)
    cls_output = outputs[1]  # batch, hidden
    cls_output = self.classifier(cls_output)  # batch, 6(classes)
    cls_output = torch.sigmoid(cls_output)  # sigmoid from logit to probability
    criterion = nn.BCELoss()  # loss function
    loss = 0
    if labels is not None:
      loss = criterion(cls_output, labels)
    return loss, cls_output

class HateTextDataset(Dataset):

    def __init__(self, tokenizer: BertTokenizer, dataframe: pd.DataFrame, lazy: bool=False):
      self.tokenizer = tokenizer
      self.pad_idx = tokenizer.pad_token_id
      self.lazy = lazy  # data conversion laziness

      if not self.lazy:
        self.X = []
        self.Y = []
        for i, (row) in tqdm(dataframe.iterrows()):  # convert data into tensor
          x, y = self.row_to_tensor(self.tokenizer, row)
          self.X.append(x)
          self.Y.append(y)
      else:
        self.df = dataframe

    @staticmethod  
    def row_to_tensor(tokenizer: BertTokenizer, row: pd.Series) -> Tuple[torch.LongTensor, torch.LongTensor]:
      tokens = tokenizer.encode(row['comment_text'], add_special_tokens=True)
      if len(tokens) > 120:
        tokens = tokens[:119] + [tokens[-1]]
      x = torch.LongTensor(tokens)
      y = torch.FloatTensor(row[['toxic', 'severe_toxic', 'obscene', 'threat',
                                 'insult', 'identity_hate']])
      return x, y

    def __len__(self):
      if self.lazy:
        return len(self.df)
      else:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
      if not self.lazy:
        return self.X[index], self.Y[index]
      else:
        return self.row_to_tensor(self.tokenizer, self.df.iloc[index])
      
def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]], device: torch.device) \
      -> Tuple[torch.LongTensor, torch.LongTensor]:
      x, y = list(zip(*batch))
      x = pad_sequence(x, batch_first=True, padding_value=0)
      y = torch.stack(y)
      return x.to(device), y.to(device)

def train(model, iterator, optimizer, scheduler):
  model.train()  # set train mode
  total_loss = 0 
  for x, y in tqdm(iterator):
    optimizer.zero_grad()  # set gradients to zero
    mask = (x != 0).float()
    loss, outputs = model(x, attention_mask=mask, labels=y)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    scheduler.step()
  print(f"Train loss {total_loss / len(iterator)}")

def evaluate(model, iterator):
  model.eval()  # set eval mode
  pred = []
  true = []
  with torch.no_grad():
    total_loss = 0
    for x, y in tqdm(iterator):
      mask = (x != 0).float()
      loss, outputs = model(x, attention_mask=mask, labels=y)
      total_loss += loss
      true += y.cpu().numpy().tolist()
      pred += outputs.cpu().numpy().tolist()
  true = np.array(true)
  pred = np.array(pred)
  for i, name in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                            'identity_hate']):
    print(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
  print(f"Evaluate loss {total_loss / len(iterator)}")

train_dataset = HateTextDataset(tokenizer, train_split_df, lazy=True)
dev_dataset = HateTextDataset(tokenizer, val_split_df, lazy=True)
collate_fn = partial(collate_fn, device=device)

BATCH_SIZE = 32
train_sampler = RandomSampler(train_dataset)
dev_sampler = RandomSampler(dev_dataset)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            sampler=train_sampler, collate_fn=collate_fn)
dev_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                          sampler=dev_sampler, collate_fn=collate_fn)

model = DEIHateClassifier(BertModel.from_pretrained(bert_model_name), 6).to(device)

no_decay = ['bias', 'LayerNorm.weight']  # no deacy parameters
optimizer_grouped_parameters = [
  {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
   'weight_decay': 0.01},
  {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
   'weight_decay': 0.0}               
]

EPOCH_NUM = 2
warmup_steps = 10 ** 3
total_steps = len(train_iterator) * EPOCH_NUM - warmup_steps
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

for i in range(EPOCH_NUM):
  print('=' * 50, f"EPOCH {i}", '=' * 50)
  train(model, train_iterator, optimizer, scheduler)
  evaluate(model, dev_iterator)

torch.save(model.state_dict(), "model_1.pt")






