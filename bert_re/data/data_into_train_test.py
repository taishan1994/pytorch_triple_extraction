# -*- coding: utf-8 -*-
import json
import re
import pandas as pd
from pprint import pprint

df = pd.read_excel('人物关系表.xlsx')
relations = list(df['关系'].unique())
relations.remove('unknown')
relation_dict = {'unknown': 0}
relation_dict.update(dict(zip(relations, range(1, len(relations)+1))))

with open('rel_dict.json', 'w', encoding='utf-8') as h:
    h.write(json.dumps(relation_dict, ensure_ascii=False, indent=2))

# print('总数: %s' % len(df))
# pprint(df['关系'].value_counts())
df['rel'] = df['关系'].apply(lambda x: relation_dict[x])

res = []
i = 1
for per1, per2, text, label in zip(df['人物1'].tolist(), df['人物2'].tolist(), df['文本'].tolist(), df['rel'].tolist()):
    # 数据有的不正确，这里修改
    if per1 == per2:
        continue
    if per1 == '黄泽胜': per1 = '黄泽生'
    if per1 == '周望第': per1 = '周望弟'
    if per1 == '李敬重': per1 = '李敬善'
    if per2 == '宋美龄。': per2 = '宋美龄'
    if per1 == '哈利王子':
        per1 = '哈里王子'
        text = text.replace('哈利王子','哈里王子')
    if per2 == '大卫*陶德':
        per2 = '大卫·陶德'
        text = text.replace('大卫*陶德','大卫·陶德')
    if per1 == '弗朗索瓦?库普兰':
        per1 = '弗朗索瓦·库普兰'
        text = text.replace('弗朗索瓦?库普兰', '弗朗索瓦·库普兰')

    # 以下是要找到实体的前后边界
    # print(i, per1, per2, text)
    # 威廉 威廉六世
    if per1 in per2:
        text_tmp = text.replace(per2, '#'*(len(per2)+2))
        text_tmp = text_tmp.replace(per1, '#'+per1+'#')
        print(text_tmp)
        text_tmp = text_tmp.replace('#'*(len(per2)+2),'$'+per2+'$')
        res1 = re.search('#'+per1+'#', text_tmp)
        res2 = re.search('\$'+per2+'\$', text_tmp)
        text = text_tmp + '\t' + str(res1.span()[0]) + '\t' + str(res1.span()[1]-1) + '\t' + str(res2.span()[0]) + '\t' + str(res2.span()[1]-1)
        print(text)
    elif per2 in per1:
        text_tmp = text.replace(per1, '#' * (len(per1) + 2))
        text_tmp = text_tmp.replace(per2, '$' + per2 + '$')
        print(text_tmp)
        text_tmp = text_tmp.replace('#' * (len(per1) + 2), '#' + per1 + '#')
        res1 = re.search('#' + per1 + '#', text_tmp)
        res2 = re.search('\$' + per2 + '\$', text_tmp)
        text = text_tmp + '\t' + str(res1.span()[0]) + '\t' + str(res1.span()[1]-1) + '\t' + str(res2.span()[0]) + '\t' + str(res2.span()[1]-1)
        print(text)
    else:
        text = text.replace(per1,'#'+per1+'#').replace(per2,'$'+per2+'$')
        res1 = re.search('#'+per1+'#', text)
        res2 = re.search('\$'+per2+'\$', text)
        text = text + '\t' + str(res1.span()[0]) + '\t' + str(res1.span()[1]-1) + '\t' + str(res2.span()[0]) + '\t' + str(res2.span()[1]-1)
    res.append([text, label])
    i += 1

df = pd.DataFrame(res, columns=['text','rel'])
df['length'] = df['text'].apply(lambda x:len(x))

# df = df.iloc[:100, :] # 取前n条数据进行模型方面的测试
# 只取文本长度小于等于128的
df = df[df['length'] <= 128]
print('总数: %s' % len(df))
pprint(df['rel'].value_counts())
# 统计文本长度分布
pprint(df['length'].value_counts())
train_df = df.sample(frac=0.8, random_state=1024)
test_df = df.drop(train_df.index)

with open('train.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(train_df['text'].tolist(), train_df['rel'].tolist()):
        f.write(str(rel)+'\t'+text+'\n')

with open('test.txt', 'w', encoding='utf-8') as g:
    for text, rel in zip(test_df['text'].tolist(), test_df['rel'].tolist()):
        g.write(str(rel)+'\t'+text+'\n')




