import os
import json
import re

re_mid_data_path = './data/dgre/re_mid_data'
mid_data_path = './data/dgre/mid_data'
train_file = mid_data_path + '/train.json'
dev_file = mid_data_path + '/dev.json'
rel_labels_file = re_mid_data_path + '/rels.txt'

if not os.path.exists(re_mid_data_path):
  os.mkdir(re_mid_data_path)

id2rellabel = {}
rellabel2id = {}
with open(rel_labels_file,'r') as fp:
  rel_labels = fp.read().strip().split('\n')
  for i,rlabel in enumerate(rel_labels):
    id2rellabel[i] = rlabel
    rellabel2id[rlabel] = i
print(rellabel2id)

def get_raw_data(output_file, input_file):
  with open(input_file,'r') as fp:
    data = json.loads(fp.read())
    total = len(data)-1
    j = 0
    for i in data:
      print(j, total)
      text = i['text']
      # 要先存一份备份
      tmp_text = text
      # print(text)
      subjects = i['subject_labels']
      objects = i['object_labels']
      tmp = []
      # print(subjects)
      # print(objects)
      # 遍历每一个主体和客体
      for sub in subjects:
        for obj in objects:
          if obj[1] in sub[1]:
            text = text[:sub[2]] + '&'*len(sub[1]) + text[sub[3]:]
            text = text[:obj[2]] + '%'*len(obj[1]) + text[obj[3]:]
            text = re.sub('&'*len(sub[1]),'#'+'&'*len(sub[1])+'#', text)
            text = re.sub('%'*len(obj[1]),'$'+'%'*len(obj[1])+'$', text)
          else:
            text = text[:obj[2]] + '%'*len(obj[1]) + text[obj[3]:]
            text = text[:sub[2]] + '&'*len(sub[1]) + text[sub[3]:] 
            text = re.sub('%'*len(obj[1]),'$'+'%'*len(obj[1])+'$', text)   
            text = re.sub('&'*len(sub[1]),'#'+'&'*len(sub[1])+'#', text)        
          try:
            sub_re = re.search('&'*len(sub[1]), text)
            sub_re_span = sub_re.span()
            sub_re_start = sub_re_span[0]
            sub_re_end = sub_re_span[1]+1
            obj_res = re.search('%'*len(obj[1]), text)
            obj_re_span = obj_res.span()
            obj_re_start = obj_re_span[0]
            obj_re_end = obj_re_span[1]+1
            text = re.sub('&'*len(sub[1]),sub[1],text)
            text = re.sub('%'*len(obj[1]),obj[1],text)
          except Exception as e:
            continue
          if sub[0] == obj[0]:
            output_file.write(sub[4] + '\t' + text + '\t' + str(sub_re_start) + '\t' +
              str(sub_re_end) + '\t' + str(obj_re_start) + '\t' + str(obj_re_end) + '\n')
          else:
            output_file.write('未知' + ' ' + text + ' ' + str(sub_re_start) + ' ' +
              str(sub_re_end) + ' ' + str(obj_re_start) + ' ' + str(obj_re_end) + '\n')
          # 恢复text
          text = tmp_text
      j+=1

if __name__ == '__main__':
  train_raw_file = open(re_mid_data_path + '/train.txt','w',encoding='utf-8')
  dev_raw_file = open(re_mid_data_path + '/dev.txt','w',encoding='utf-8')
  get_raw_data(train_raw_file, train_file)
  get_raw_data(dev_raw_file, dev_file)
  train_raw_file.close()
  dev_raw_file.close()
