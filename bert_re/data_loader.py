import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
try:
  from utils.utils import sequence_padding
except:
  from .utils.utils import sequence_padding


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path



# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        examples = []
        with open(filename, encoding='utf-8') as f:
            raw_examples = f.readlines()
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
          # print(i,item)
          item = item.strip().split('\t')
          text = item[1]
          labels = item[0]
          ids = item[2:6]
          examples.append((text, labels, ids))  # 注意，这里的ids里面的索引已经加上了CLS
        return examples

class Collate:
  def __init__(self, max_len, tag2id, device, tokenizer):
      self.maxlen = max_len
      self.tag2id = tag2id
      self.id2tag = {v:k for k,v in tag2id.items()}
      self.device = device
      self.tokenizer = tokenizer

  def collate_fn(self, batch):

      batch_labels = []
      batch_ids = []
      batch_token_ids = []
      batch_attention_mask = []
      batch_token_type_ids = []
      callback = []
      for i, (text, label ,ids) in enumerate(batch):
          if len(text) == 0:
            continue
          if len(text) > self.maxlen - 2:
            text = text[:self.maxlen - 2]
          tokens = [i for i in text]
          tokens = ['[CLS]'] + tokens + ['[SEP]']
          # 过滤掉超过文本最大长度的
          flag = False
          for j in ids:
            if int(j) > self.maxlen - 2:
              flag = True
              break
          if flag:
            continue
          token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
          tmp_length = self.maxlen - len(token_ids)
          batch_attention_mask.append([1] * len(token_ids) + [0] * tmp_length)
          batch_token_type_ids.append([0] * self.maxlen)
          token_ids = token_ids + [0] * tmp_length
          batch_token_ids.append(token_ids)  # 前面已经限制了长度
          batch_labels.append(int(self.tag2id[label]))
          batch_ids.append([int(m) for m in ids])
          callback.append((text, label))
       
      # batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      # attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=self.maxlen), dtype=torch.long, device=self.device)
      # token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
      attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
      token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
      batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
      batch_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)

      return batch_token_ids, attention_mask, token_type_ids, batch_labels, batch_ids, callback


if __name__ == "__main__":
  from transformers import BertTokenizer
  max_len = 300
  tokenizer = BertTokenizer.from_pretrained('../model_hub/chinese-bert-wwm-ext/vocab.txt')
  train_dataset = MyDataset(file_path='../data/re_mid_data/train.txt')
  # print(train_dataset[0])

  with open('../data/re_mid_data/rels.txt', 'r') as fp:
    labels = fp.read().split('\n')
  id2tag = {}
  tag2id = {}
  for i,label in enumerate(labels):
    id2tag[i] = label
    tag2id[label] = i
  print(tag2id)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  collate = Collate(max_len=max_len, tag2id=tag2id, device=device, tokenizer=tokenizer)
  collate.collate_fn(train_dataset[:16])
  batch_size = 2
  train_dataset = train_dataset[:10]
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate.collate_fn) 

  for i, batch in enumerate(train_dataloader):
    leng = len(batch) - 1
    for j in range(leng):
      print(batch[j].shape)
    break
