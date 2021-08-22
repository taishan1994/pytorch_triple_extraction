import json
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
# 这里要显示的引入BertFeature，不然会报错
try:
  from preprocess import BertFeature
  from preprocess import get_out, Processor
  import bert_config
except Exception as e:
  from .preprocess import BertFeature
  from .preprocess import get_out, Processor
  from . import bert_config


class ReDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)

        self.token_ids = (torch.tensor(example.token_ids).long() for example in features)
        self.attention_masks = (torch.tensor(example.attention_masks).float() for example in features)
        self.token_type_ids = (torch.tensor(example.token_type_ids).long() for example in features)
        self.labels = (torch.tensor(example.labels).long() for example in features)
        self.ids = (torch.tensor(example.ids).long() for example in features)

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': next(self.token_ids),
            'attention_masks': next(self.attention_masks),
            'token_type_ids': next(self.token_type_ids),
        }

        data['labels'] = next(self.labels)
        data['ids'] = next(self.ids)

        return data

if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    args.log_dir = './logs/'
    args.max_seq_len = 300
    args.bert_dir = '../model_hub/chinese-roberta-wwm-ext/'

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('../data/re_mid_data/rels.txt','r') as fp:
        labels = fp.read().strip().split('\n')
    for i,j in enumerate(labels):
        label2id[j] = i
        id2label[i] = j
    print(label2id)

    # train_out = get_out(processor, './data/train.txt', args, id2label, 'train')
    # dev_out = get_out(processor, './data/test.txt', args, id2label, 'dev')
    # test_out = get_out(processor, './data/test.txt', args, id2label, 'test')

    import pickle
    
    train_out = pickle.load(open('../data/re_final_data/train.pkl','rb'))
    train_features, train_callback_info = train_out
    train_dataset = ReDataset(train_features)
    for data in train_dataset:
        print(data['token_ids'])
        print(data['attention_masks'])
        print(data['token_type_ids'])
        print(data['labels'])
        print(data['ids'])
        break

    args.train_batch_size = 2
    train_dataset = ReDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)
    for step, train_data in enumerate(train_loader):
        print(train_data['token_ids'].shape)
        print(train_data['attention_masks'].shape)
        print(train_data['token_type_ids'].shape)
        print(train_data['labels'])
        print(train_data['ids'])
        break