import bert_bilstm_crf_ner.config as ner_config
import bert_bilstm_crf_ner.bert_ner_model as ner_model
import bert_bilstm_crf_ner.main as ner_main
import bert_re.main as re_main
import bert_re.bert_config as re_config
import bert_re.models as re_model

import os
import re
import logging 
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

def get_ner_result(raw_text):
  # 命名实体识别相关
  model_name = 'bert_crf'
  ner_args = ner_config.Args().get_parser()
  ner_args.bert_dir = './model_hub/chinese-roberta-wwm-ext/'
  ner_args.gpu_ids = "-1"
  ner_args.use_lstm = 'False'
  ner_args.use_crf = 'True'
  ner_args.num_tags = 5
  ner_args.max_seq_len = 512
  ner_args.num_layers = 1
  ner_args.lstm_hidden = 128
  nerlabel2id = {}
  id2nerlabel = {}
  with open('./data/dgre/mid_data/ner_labels.txt','r') as fp:
      ner_labels = fp.read().strip().split('\n')
  for i,j in enumerate(ner_labels):
    nerlabel2id[j] = i
    id2nerlabel[i] = j
  logger.info(id2nerlabel)
  bertForNer = ner_main.BertForNer(ner_args, None, None, None, id2nerlabel)
  model_path = './bert_bilstm_crf_ner/checkpoints/{}/model.pt'.format(model_name)
  pred_entities = bertForNer.predict(raw_text, model_path)
  return pred_entities

def get_re_result(entities, raw_text):
  # 首先先区分是主体还是客体
  subjects = []
  objects = []
  for info in entities:
    print(info)
    if info[2] == 'subject':
      subjects.append((info[0],info[1],info[1]+len(info[0])))
    elif info[2] == 'object':
      objects.append((info[0],info[1],info[1]+len(info[0])))
  print(subjects)
  print(objects)
  re_args = re_config.Args().get_parser()
  re_args.bert_dir = './model_hub/chinese-roberta-wwm-ext/'
  re_args.gpu_ids = "-1"
  re_args.num_tags = 5
  re_args.max_seq_len = 512
  trainer = re_main.Trainer(re_args, None, None, None)
  re_args.output_dir = './bert_re/checkpoints/'
  tokenizer = BertTokenizer.from_pretrained(re_args.bert_dir)
  process_data = transforme_re_data(subjects, objects, raw_text)
  label2id = {}
  id2label = {}
  with open('./data/dgre/re_mid_data/rels.txt','r') as fp:
      labels = fp.read().strip().split('\n')
  for i,j in enumerate(labels):
      label2id[j] = i
      id2label[i] = j
  for data in process_data:
    relation = trainer.predict(tokenizer, data[0], id2label, re_args, data[1])
    print("==========================")
    print(raw_text)
    print("主体：", data[2][0])
    print("客体：", data[2][1])
    print("关系：", "".join(relation))

def transforme_re_data(subjects, objects, text):
  # 遍历每一个主体和客体
  tmp_text = text
  process_data = []
  for sub in subjects:
    for obj in objects:
      if obj[0] in sub[0]:
        text = text[:sub[1]] + '&'*len(sub[0]) + text[sub[2]:]
        text = text[:obj[1]] + '%'*len(obj[0]) + text[obj[2]:]
        text = re.sub('&'*len(sub[0]),'#'+'&'*len(sub[0])+'#', text)
        text = re.sub('%'*len(obj[0]),'$'+'%'*len(obj[0])+'$', text)
      else:
        text = text[:obj[1]] + '%'*len(obj[0]) + text[obj[2]:]
        text = text[:sub[1]] + '&'*len(sub[0]) + text[sub[2]:] 
        text = re.sub('%'*len(obj[0]),'$'+'%'*len(obj[0])+'$', text)   
        text = re.sub('&'*len(sub[0]),'#'+'&'*len(sub[0])+'#', text)      
      try:
        sub_re = re.search('&'*len(sub[0]), text)
        sub_re_span = sub_re.span()
        sub_re_start = sub_re_span[0]
        sub_re_end = sub_re_span[1]+1
        obj_res = re.search('%'*len(obj[0]), text)
        obj_re_span = obj_res.span()
        obj_re_start = obj_re_span[0]
        obj_re_end = obj_re_span[1]+1
        text = re.sub('&'*len(sub[0]),sub[0],text)
        text = re.sub('%'*len(obj[0]),obj[0],text)
      except Exception as e:
        print(e)
        continue
      process_data.append((text,[sub[1],sub[2],obj[1],obj[2]],(sub,obj)))
      # 恢复text
      text = tmp_text
  return process_data


if __name__ == '__main__':

  raw_texts = [
    '明早起飞》是由明太鱼作词，满江作曲，戴娆演唱的一首歌曲',
    '古董相机收藏与鉴赏》是由高继生、高峻岭编著，浙江科学技术出版社出版的一本书籍',
    '谢顺光，男，祖籍江西都昌，出生于景德镇陶瓷世家',
  ]

  raw_texts = [
    '故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失',
    '1045号汽车故障报告故障现象打开点火开关，操作左前电动座椅开关，座椅6个方向均不动作故障原因六向电动座椅线束磨破搭铁修复方法包扎磨破线束，从新固定。',
  ]
  for raw_text in raw_texts:
    entities = get_ner_result(raw_text)
    get_re_result(entities, raw_text)
