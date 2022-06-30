# pytorch_triplet_extraction
延申：
- 基于GlobalPointer的三元组抽取，又快又准确：https://github.com/taishan1994/pytorch_GlobalPointer_triple_extraction
- 基于casrel的三元组抽取，使用更加方便：https://github.com/taishan1994/pytorch_casrel_triple_extraction
****
- 一种基于globalpointer的命名实体识别：https://github.com/taishan1994/pytorch_GlobalPointer_Ner
- 一种基于TPLinker_plus的命名实体识别：https://github.com/taishan1994/pytorch_TPLinker_Plus_Ner
- 一种one vs rest方法进行命名实体识别：https://github.com/taishan1994/pytorch_OneVersusRest_Ner
- 一种级联Bert用于命名实体识别，解决标签过多问题：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner
- 一种多头选择Bert用于命名实体识别：https://github.com/taishan1994/pytorch_Multi_Head_Selection_Ner
- 中文命名实体识别最新进展：https://github.com/taishan1994/awesome-chinese-ner
- 信息抽取三剑客：实体抽取、关系抽取、事件抽取：https://github.com/taishan1994/chinese_information_extraction
- 一种基于机器阅读理解的命名实体识别：https://github.com/taishan1994/BERT_MRC_NER_chinese
- W2NER：命名实体识别最新sota：https://github.com/taishan1994/W2NER_predict
****
基于pytorch的中文三元组提取（命名实体识别+关系抽取）<br>
预训练模型为<a href='https://huggingface.co/hfl/chinese-roberta-wwm-ext'>chinese-roberta-wwm-ext</a><br>
训练好的命名实体识别模型：<br>
链接：https://pan.baidu.com/s/1ZrC4eum6cR8_UZZI9vxzFg <br>
提取码：68wg <br>
训练好的关系抽取模型：<br>
链接：https://pan.baidu.com/s/1HIf6ri0BLv3Aeu20o_lwGg <br>
提取码：7cee<br>
data下面的数据：<br>
链接：https://pan.baidu.com/s/15v8SxWpzQ5HwjXETxWnnhg <br>
提取码：g53x<br>
由于关系抽取数据量有点大，只以batchsize=16运行了4000个step。<br>
具体命名实体识别和关系抽取在相应的readme.md里面有细讲。<br>

# 说明
命名实体识别基于bert_bilstm_crf，识别出句子中的主体(subject)和客体(object)。相关功能在bert_bilstm_crf_ner下。存储的模型在bert_bilstm_crf_ner/checkpoints/bert_bilsm_crf/model.pt<br>
关系抽取基于bert，识别出主体和客体之间的关系。相关功能在bert_re下。存储的模型位于bert_re/checkpoints/best.pt<br>
具体相关的数据位于/data/下面，可以去查看。<br>

# 命名实体识别
在bert_bilstm_crf_ner文件夹下的main.py是主运行程序，可用以下命令运行训练测试和预测：<br>
```python
!python main.py \
--bert_dir="../model_hub/chinese-roberta-wwm-ext/" \
--data_dir="../data/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=5 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=300 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=1 \
--eval_batch_size=32 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='True' \
--use_crf='True' \
--dropout_prob=0.3 \
--dropout=0.3 \
```
```python
2021-08-18 12:17:05,417 - INFO - main.py - test - 144 -               
               precision    recall  f1-score   support

      object       0.76      0.89      0.82     38656
     subject       0.76      0.85      0.80     25103

   micro avg       0.76      0.88      0.81     63759
   macro avg       0.76      0.87      0.81     63759
weighted avg       0.76      0.88      0.81     63759
```

# 关系抽取结果
在bert_re文件夹下的main.py是主运行程序，可用以下命令运行训练测试和预测：<br>
```python
!python main.py \
--bert_dir="../model_hub/chinese-roberta-wwm-ext/" \
--data_dir="../data/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=49 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=300 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=16 \
--train_epochs=1 \
--eval_batch_size=32 \
--dropout_prob=0.3 \
```
```python
2021-08-22 05:22:50,141 - INFO - main.py - <module> - 247 - 【test】 loss：631.809930 accuracy：0.8841 micro_f1：0.8841 macro_f1：0.8720
2021-08-22 05:22:50,292 - INFO - main.py - <module> - 249 -               precision    recall  f1-score   support

          歌手       0.83      0.82      0.83      3961
         代言人       0.98      1.00      0.99       920
          作曲       0.55      0.71      0.62      1647
          父亲       0.92      0.64      0.76      3036
        占地面积       0.93      0.93      0.93        90
        注册资本       1.00      1.00      1.00       116
        所属专辑       0.96      0.96      0.96      1063
        所在城市       0.95      0.98      0.97       128
          首都       1.00      1.00      1.00        82
        毕业院校       1.00      0.99      1.00      1184
          饰演       0.94      0.95      0.95      1420
          祖籍       0.99      0.93      0.96       206
        上映时间       0.96      1.00      0.98       886
          主角       0.86      0.67      0.75       350
           号       0.70      0.98      0.82       258
          简称       1.00      0.99      0.99       588
          校长       0.97      1.00      0.98       414
          丈夫       0.64      0.92      0.76      2866
          国籍       0.99      0.98      0.98      2030
          导演       0.88      0.95      0.91      2957
         主题曲       0.97      0.95      0.96       486
        专业代码       0.00      0.00      0.00         2
          妻子       0.81      0.91      0.86      2869
        官方语言       1.00      1.00      1.00        24
        成立日期       1.00      1.00      1.00      2082
          配音       0.99      0.95      0.97       586
        邮政编码       1.00      1.00      1.00         2
          海拔       1.00      1.00      1.00        54
          作词       0.82      0.52      0.63      1669
         创始人       0.84      0.86      0.85       308
         主持人       0.86      0.89      0.87       696
          母亲       0.98      0.52      0.68      1661
        人口数量       1.00      0.94      0.97        64
        修业年限       0.00      0.00      0.00         2
         制片人       0.79      0.70      0.74       248
          编剧       0.82      0.56      0.67       902
          气候       1.00      0.98      0.99       104
         改编自       0.95      0.98      0.96       108
          票房       1.00      1.00      1.00       296
          主演       0.96      0.96      0.96      7056
          面积       0.92      0.92      0.92        76
        出品公司       0.98      0.98      0.98      1056
          朝代       0.96      0.99      0.97      1026
         董事长       0.98      0.94      0.96      1220
          作者       0.93      0.96      0.94      4260
          嘉宾       0.77      0.87      0.82       922
          获奖       1.00      1.00      1.00       584
        总部地点       1.00      0.99      0.99       408

    accuracy                           0.88     52973
   macro avg       0.88      0.87      0.87     52973
weighted avg       0.89      0.88      0.88     52973
```

# 融合预测
在得到了各自的模型之后，在get_result.py中可以进行三元组抽取了：
```
python get_result.py
```
```
('明早起飞', 0, 'subject')
('明太鱼', 7, 'object')
('满江', 13, 'object')
('戴娆', 18, 'object')
[('明早起飞', 0, 4)]
[('明太鱼', 7, 10), ('满江', 13, 15), ('戴娆', 18, 20)]
==========================
明早起飞》是由明太鱼作词，满江作曲，戴娆演唱的一首歌曲
主体： ('明早起飞', 0, 4)
客体： ('明太鱼', 7, 10)
关系： 作词
==========================
明早起飞》是由明太鱼作词，满江作曲，戴娆演唱的一首歌曲
主体： ('明早起飞', 0, 4)
客体： ('满江', 13, 15)
关系： 作曲
==========================
明早起飞》是由明太鱼作词，满江作曲，戴娆演唱的一首歌曲
主体： ('明早起飞', 0, 4)
客体： ('戴娆', 18, 20)
关系： 作曲

('古董相机收藏与鉴赏', 0, 'subject')
('高继生', 12, 'object')
('高峻岭', 16, 'object')
[('古董相机收藏与鉴赏', 0, 9)]
[('高继生', 12, 15), ('高峻岭', 16, 19)]
==========================
古董相机收藏与鉴赏》是由高继生、高峻岭编著，浙江科学技术出版社出版的一本书籍
主体： ('古董相机收藏与鉴赏', 0, 9)
客体： ('高继生', 12, 15)
关系： 作者
==========================
古董相机收藏与鉴赏》是由高继生、高峻岭编著，浙江科学技术出版社出版的一本书籍
主体： ('古董相机收藏与鉴赏', 0, 9)
客体： ('高峻岭', 16, 19)
关系： 作者

('谢顺光', 0, 'subject')
('江西都昌', 8, 'object')
[('谢顺光', 0, 3)]
[('江西都昌', 8, 12)]
==========================
谢顺光，男，祖籍江西都昌，出生于景德镇陶瓷世家
主体： ('谢顺光', 0, 3)
客体： ('江西都昌', 8, 12)
关系： 祖籍
```
# 存在的问题
有很多可以改进的地方：<br>
- 一个实体可能是主体，也可能是客体，这里没有考虑到。实际上可以先识别出各种类型的实体，而不是定义为主体和客体。（或许我们先找出所有的实体，然后根据实体间的关系，反过来推测实体的类型。）
- 关系抽取，每次都需要将一对实体和文本输入到网络中进行预测，可以试下输入多个实体对一次性进行多分类，避免对一个句子进行重复编码。
- 没有未知这一类，对于不在关系类别中的没有办法。（或许可以在预测的时候设置一个阈值，如果大于该阈值才认定为那一类，否则就是未知类）

# 参考
借鉴了该博客的思想：https://www.cnblogs.com/jclian91/p/12499062.html （基于keras）<br>

# 补充
更为简单的三元组抽取，联合关系抽取，不用再先进行实体识别，再进行关系分类：
- https://github.com/taishan1994/OneRel_chinese <br>
- [信息抽取三剑客：实体抽取、关系抽取、事件抽取](https://github.com/taishan1994/chinese_information_extraction)
