import os
import pickle
import logging
import codecs
from transformers import BertTokenizer
try:
  import bert_config
  from utils import utils
except Exception as e:
  from . import bert_config
  from .utils import utils
import numpy as np
import json


logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None, ids=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.ids = ids


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None, ids=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels
        # ids
        self.ids = ids

class Processor:

    @staticmethod
    def read_txt(file_path):
        with codecs.open(file_path,'r',encoding='utf-8') as f:
            raw_examples = f.read().strip()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        # 这里是从json数据中的字典中获取
        for line in raw_examples.split('\n'):
            line = line.split('\t')
            if len(line) == 6:
                labels = line[0]
                text = line[1]
                ids = [int(line[2]),int(line[3]),int(line[4]),int(line[5])]
                examples.append(InputExample(set_type=set_type,
                                         text=text,
                                         labels=labels,
                                         ids=ids))
        return examples


def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer, max_seq_len, label2id):
    set_type = example.set_type
    raw_text = example.text
    labels = example.labels
    ids =example.ids
    # 文本元组
    callback_info = (raw_text,)
    callback_labels = label2id[labels]
    callback_info += (callback_labels,)
    labels = label2id[labels]

    # label_ids = label2id[labels]
    ids = [x for x in ids]
    tokens = [i for i in raw_text]
    encode_dict = tokenizer.encode_plus(text=tokens,
                                        add_special_tokens=True,
                                        max_length=max_seq_len,
                                        truncation='longest_first',
                                        padding="max_length",
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3:
        decode_text = tokenizer.decode(np.array(token_ids)[np.where(np.array(attention_masks) == 1)[0]].tolist())
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f"text: {decode_text}")
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"labels: {labels}")
        logger.info(f"ids：{ids}")

    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=labels,
        ids=ids
    )

    return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, bert_dir, label2id):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')
    longer_count = 0
    for i, example in enumerate(examples):
        ids = example.ids
        flag = False
        for x in ids:
            if x > max_seq_len - 1:
                longer_count += 1
                flag = True
                break
        if flag:
            continue
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            label2id=label2id,
        )
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')
    logger.info(f"超出最大长度的有：{longer_count}")
    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_out(processor, txt_path, args, id2label, label2id, mode):
    raw_examples = processor.read_txt(txt_path)

    examples = processor.get_examples(raw_examples, mode)
    for i, example in enumerate(examples):
        print("==========================")
        print(example.text)
        print(example.labels)
        print(example.ids)
        print("==========================")
        if i == 5:
            break
    out = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, label2id)
    def save_pkl(data_dir, data, desc):
      """保存.pkl文件"""
      with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'wb') as f:
          pickle.dump(data, f)
    save_path = os.path.join(args.data_dir, 're_final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_pkl(save_path, out, mode)
    return out


if __name__ == '__main__':
    data_name = "dgre"

    args = bert_config.Args().get_parser()
    args.log_dir = './logs/'
    args.bert_dir = '../model_hub/chinese-roberta-wwm-ext/'
    utils.set_logger(os.path.join(args.log_dir, 'preprocess.log'))
    logger.info(vars(args))

    if data_name == "dgre":
        args.max_seq_len = 512
        args.data_dir = '../data/dgre/'
        re_mid_data_path = '../data/dgre/re_mid_data'

    elif data_name == "duie":
        args.max_seq_len = 300
        re_mid_data_path = '../data/re_mid_data'

    processor = Processor()

    label2id = {}
    id2label = {}
    with open(re_mid_data_path+'/rels.txt','r') as fp:
        labels = fp.read().split('\n')
    for i,j in enumerate(labels):
        label2id[j] = i
        id2label[i] = j
    print(label2id)
    train_out = get_out(processor, re_mid_data_path+'/train.txt', args, id2label, label2id, 'train')
    dev_out = get_out(processor, re_mid_data_path+'/dev.txt', args, id2label, label2id, 'dev')
    test_out = get_out(processor, re_mid_data_path+'/dev.txt', args, id2label, label2id, 'test')
