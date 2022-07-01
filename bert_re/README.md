# pytorch_bert_relation_extraction
****
### 这里使用的数据和外面的不太一样，以外面的说明为主
****
基于pytorch+bert的中文关系抽取<br>
该项目针对于中文实体对进行关系抽取，例如：<br>
2	曹操南征荆州，#刘表#之子$刘琮$投降，刘备领军民十余万避难，于当阳遭遇曹军追兵，惨败。	7	10	13	16<br>
其中：<br>
```
- 2：关系类别<br>
- 文本：曹操南征荆州，#刘表#之子$刘琮$投降，刘备领军民十余万避难，于当阳遭遇曹军追兵，惨败。<br>
- 7	10	13	16：7、10是第一个实体左右的#号在文本出现的位置（第一次），13、16是第二个实体左右的$在文本中出现的位置（第一次）。<br>
```
我们在预处理的时候会将实体一第一次出现的位置用：'#'+实体名1+'#'代替，将实体二第一次出现的位置用：'$'+实体名2+'$'代替。在经过bert进行编码后，取得#和$表示的嵌入，进行拼接后经过全连接层进行关系分类。<br>

# 文件说明
--logs：存放日志<br>
--checkpoints：存放保存的模型<br>
--data：存放数据<br>
--utils：存放辅助函数<br>
--bert_config.py：相关配置<br>
--dataset.py：制作数据为torch所需的格式<br>
--preprocess.py：数据预处理成bert所需要的格式<br>
--models.py：存放模型代码<br>
--main.py：主运行程序，包含训练、验证、测试、预测以及相关评价指标的计算<br>
要预先下载好预训练的bert模型，放在和该项目同级下的model_hub文件夹下，即：<br>
model_hub/bert-base-chinese/
相关下载地址：<a href="https://huggingface.co/bert-base-chinese/tree/main=">bert-base-chinese</a><br>
需要的是vocab.txt、config.json、pytorch_model.bin<br>
你也可以使用我已经训练好的模型，将其放在checkpoints下：<br>
链接：https://pan.baidu.com/s/11a-GwesO8uQMmXSYLf4FHA <br>
提取码：lsz8<br>

# 其他细节
1、由于bert会在句子前面添加一个[cls],因此在实际使用时，#和$的位置都要+1。<br>
2、会有爱德华、爱德华六世这种一种实体是另一种的一部分的情况，在预处理的时候要特殊处理，避免将爱德华六世变为#爱德华#六世，具体可以参考preprocess.py。<br>
3、在models.py中使用index_select获取#和$所对应的表示，并使用torch.cat进行拼接。<br>
4、我们选择长度小于等于128的句子。

# 运行代码
```python
python main.py \
--bert_dir="../model_hub/bert-base-chinese/" \
--data_dir="./data/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=14 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=128 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=5 \
--eval_batch_size=32 \
```

# 结果
## 训练和验证
```python
2021-08-04 15:35:09,470 - INFO - main.py - train - 75 - 【train】 epoch：4 step:399/470 loss：0.114712
2021-08-04 15:35:11,669 - INFO - main.py - train - 81 - 【dev】 loss：13.144311 accuracy：0.8471 micro_f1：0.8471 macro_f1：0.8480
2021-08-04 15:35:11,670 - INFO - main.py - train - 83 - ------------>保存当前最好的模型
```

## 测试
```python
========进行测试========
2021-08-04 15:35:36,837 - INFO - main.py - <module> - 230 - 【test】 loss：13.144311 accuracy：0.8471 micro_f1：0.8471 macro_f1：0.8480
2021-08-04 15:35:36,847 - INFO - main.py - <module> - 232 -               
                precision    recall  f1-score   support

     unknown       0.78      0.81      0.80       197
          夫妻       0.88      0.84      0.86        69
          父母       0.92      0.90      0.91       132
        兄弟姐妹       0.81      0.92      0.86        38
         上下级       0.83      0.62      0.71        32
          师生       0.82      0.86      0.84        42
          好友       0.90      0.75      0.82        24
          同学       0.96      0.90      0.93        29
          合作       0.83      0.92      0.88        53
          同人       0.94      0.87      0.91        39
          情侣       0.95      0.74      0.83        27
          祖孙       0.86      0.86      0.86        22
          同门       0.84      0.95      0.89        22
          亲戚       0.71      0.85      0.77        26

    accuracy                           0.85       752
   macro avg       0.86      0.84      0.85       752
weighted avg       0.85      0.85      0.85       752
```

## 预测
```python
2021-08-04 15:56:31,480 - INFO - main.py - <module> - 247 - 1月1日，有法制官方账号在短视频平台透露了#李双江#之子$李天一$的近况。
2021-08-04 15:56:32,063 - INFO - main.py - <module> - 249 - 预测标签：父母
2021-08-04 15:56:32,063 - INFO - main.py - <module> - 250 - 真实标签：父母
2021-08-04 15:56:32,063 - INFO - main.py - <module> - 251 - ==========================
2021-08-04 15:56:32,063 - INFO - main.py - <module> - 247 - 幼年家贫，只能参加贵格会的学校，富裕的教师#鲁宾孙#很喜欢$道尔顿$，允许他阅读自己的书和期刊。
2021-08-04 15:56:32,644 - INFO - main.py - <module> - 249 - 预测标签：unknown
2021-08-04 15:56:32,644 - INFO - main.py - <module> - 250 - 真实标签：师生
2021-08-04 15:56:32,644 - INFO - main.py - <module> - 251 - ==========================
2021-08-04 15:56:32,644 - INFO - main.py - <module> - 247 - 1537年，#亨利八世#和他的第三个王后$简·西摩$生了一个男孩：爱德华（后来的爱德华六世）。
2021-08-04 15:56:33,229 - INFO - main.py - <module> - 249 - 预测标签：夫妻
2021-08-04 15:56:33,229 - INFO - main.py - <module> - 250 - 真实标签：夫妻
2021-08-04 15:56:33,229 - INFO - main.py - <module> - 251 - ==========================
2021-08-04 15:56:33,229 - INFO - main.py - <module> - 247 - 1544年，伊丽莎白年迈的父亲#亨利#最终娶了在宫廷任职的$凯瑟琳·帕尔$为他的第六任妻子。
2021-08-04 15:56:33,811 - INFO - main.py - <module> - 249 - 预测标签：夫妻
2021-08-04 15:56:33,811 - INFO - main.py - <module> - 250 - 真实标签：夫妻
2021-08-04 15:56:33,811 - INFO - main.py - <module> - 251 - ==========================
```

# 参考
数据集来源：https://www.cnblogs.com/jclian91/p/12328570.html<br>
思路参考keras代码：https://github.com/shifop/people_relation_extract/blob/main/model.py


