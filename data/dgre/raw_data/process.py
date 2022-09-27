import json
import codecs
from pprint import pprint

data_path = "train.json"

with codecs.open(data_path, 'r', encoding="utf-8") as fp:
    data = fp.readlines()


res = []
for did, d in enumerate(data):
    d = eval(d)
    tmp = {}
    tmp["id"] = d['ID']
    tmp['text'] = d['text']
    tmp['subject_labels'] = []
    tmp['object_labels'] = []
    ent_id = 0
    for rel_id,spo in enumerate(d['spo_list']):
        h = spo['h']
        h['pos'][0]
        tmp['subject_labels'].append(["T{}".format(ent_id), d['text'][h['pos'][0]:h['pos'][1]], h['pos'][0], h['pos'][1], spo["relation"]])
        t = spo['t']
        tmp['object_labels'].append(["T{}".format(ent_id), d['text'][t['pos'][0]:t['pos'][1]], t['pos'][0], t['pos'][1], spo["relation"]])
        ent_id += 1
    res.append(tmp)

ratio = 0.8
length = len(res)
train_data = res[:int(length*ratio)]
dev_data = res[int(length*ratio):]
with open('../mid_data/train.json', 'w', encoding='utf-8') as fp:
  fp.write(json.dumps(train_data, ensure_ascii=False))
with open('../mid_data/dev.json', 'w', encoding='utf-8') as fp:
  fp.write(json.dumps(dev_data, ensure_ascii=False))