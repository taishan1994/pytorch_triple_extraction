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
**2022-09-26：保姆级教程来了！！！**

这里以[工业知识图谱关系抽取-高端装备制造知识图谱自动化构建 竞赛 - DataFountain](https://www.datafountain.cn/competitions/584)为例，一步一步的进行。

- 拷贝项目：```git clone https://github.com/taishan1994/pytorch_triple_extraction.git ```。

- 下载预训练模型[chinese-roberta-wwm-ext]([hfl/chinese-roberta-wwm-ext at main (huggingface.co)](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main))到model_hub/chinese-roberta-wwm-ext/下，需要的是config.json、pytorch_model.bin和vocab.txt，当然也可以去下载[chinese-bert-wwm-ext]([hfl/chinese-bert-wwm-ext at main (huggingface.co)](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main))到model_hub/chinese-bert-wwm-ext/下。

- 在data下新建一个数据集文件夹，针对该数据集是dgre，在dgre下新建好相应的一些文件夹，目录如下（文件夹内具体文件稍后再按步骤生成）：
	```python
	data
	----dgre
	--------mid_data  # 运行raw_data下的process.py后得到
	----------------dev.json
	----------------ent_labels.txt  # 实体名，这里就两种：subject和object
	----------------ner_labels.txt  # 字符标签，BIO格式，共五种。
	----------------train.json
	--------ner_final_data  # 在 bert_bilstm_crf_ner下运行preprocess.py后获得
	----------------dev.pkl  
	----------------train.pkl
	--------raw_data  # 原始数据
	----------------evalA.json
	----------------process.py  # 将数据处理得到mid_data下的train.json和devjson
	----------------train.json
	--------re_final_data  # 在bert_re下运行prerocess.py后获得（数据量太大会有问题，后面舍弃了，改用data_loader.py）
	----------------dev.pkl
	----------------test.pkl
	----------------train.pkl
	--------re_mid_data  # 运行re_process.py后获得
	----------------dev.txt
	----------------rels.txt  # 关系类别
	----------------train.txt
	--------dgre_512_cut.txt  # 实体识别处理后可视化结果
	```

	raw_data文件夹用于存储原始的数据。该竞赛原始数据由两部分组成，evalA.json（只有文本，没有标签）和train.json（训练数据），train.json里面每一行是一个字典，我们看看单条数据：

	```python
	{"ID": "AT0001", "text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。", "spo_list": [{"h": {"name": "发动机", "pos": [28, 31]}, "t": {"name": "熄火", "pos": [31, 33]}, "relation": "部件故障"}]}
	```

	"h"表示关系主体，"t"表示关系客体，"relation"表示关系。在raw_data下新建一个process.py，该文件主要是将数据处理成之后我们需要的格式，在mid_data下这里看看处理完之后的数据是什么样子（由于只有train.json，因此我们需要对数据划分为训练集和验证集）：

	```python
	[{"id": "AT0001", "text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。", "subject_labels": [["T0", "发动机", 28, 31, "部件故障"]], "object_labels": [["T0", "熄火", 31, 33, "部件故障"]]}, ...] 
	```

	需要注意两个地方，每一个实体列表5项分别表示：[ID标识，实体，实体起始位置，实体结束位置，关系]，subject_labels表示主体实体，object_labels表示客体实体，主体实体和客体实体之间通过ID标识连接。ent_labels.txt和ner_labels.txt自己新建然后输入以下信息就行：

	```python
	ent_labels.txt里：
	subject
	object
	
	ner_labels.txt里：
	O
	B-object
	I-object
	B-subject
	I-subject
	```

- 接下来我们解可以进行实体识别提取主体和客体了：
	```python
	cd bert_bilstm_crf_ner
	在preprocess.py里面我们需要修改以下一些地方：
	dataset = "dgre"
	
	if dataset == "dgre":
	    args.data_dir = '../data/dgre/'  # 数据集地址
	    args.max_seq_len = 512  # 文本最大长度
	
	对于一个新的数据集，我们只需要修改dataset为我们data的名字，并新建一个if-else分支，指定数据目录和文本最大长度。
	最后运行python preprocess.py即可获得ner_final_data下数据。
	
	接着在main.py里面修改data_name为"dgre"，新建一个if-else分支用于输入预测文本：
	if data_name == "dgre":
	    raw_text = "211号汽车故障报告综合情况:故障现象:开暖风鼓风机运转时有异常响声。故障原因简要分析:该故障是鼓风机运转时有异响由此可以判断可能原因：1鼓风机故障 2鼓风机内有杂物"
	    
	最后通过以下指令训练，验证，测试和预测（输入指令时把后面注释给删掉）：
	python main.py \
	--bert_dir="../model_hub/chinese-bert-wwm-ext/" \  # 预训练模型名称
	--data_dir="../data/dgre/" \
	--log_dir="./logs/" \
	--output_dir="./checkpoints/" \
	--num_tags=5 \
	--seed=123 \
	--gpu_ids="0" \
	--max_seq_len=512 \  # 和preprocess.py里面的一致
	--lr=3e-5 \
	--crf_lr=3e-2 \
	--other_lr=3e-4 \
	--train_batch_size=8 \
	--train_epochs=5 \
	--eval_batch_size=8 \
	--max_grad_norm=1 \
	--warmup_proportion=0.1 \
	--adam_epsilon=1e-8 \
	--weight_decay=0.01 \
	--lstm_hidden=128 \
	--num_layers=1 \
	--use_lstm="False" \
	--use_crf="True" \
	--dropout_prob=0.3 \
	--dropout=0.3 
	
	结果：
	              precision    recall  f1-score   support
	
	      object       0.67      0.78      0.72      1201
	     subject       0.68      0.79      0.73      1177
	
	   micro avg       0.67      0.78      0.72      2378
	   macro avg       0.67      0.78      0.72      2378
	weighted avg       0.67      0.78      0.72      2378
	
	211号汽车故障报告综合情况:故障现象:开暖风鼓风机运转时有异常响声。故障原因简要分析:该故障是鼓风机运转时有异响由此可以判断可能原因：1鼓风机故障 2鼓风机内有杂物
	[('鼓风机', 23, 'subject'), ('有异常响声', 29, 'object'), ('鼓风机', 48, 'subject'), ('异响', 55, 'object'), ('鼓风机', 69, 'subject'), ('故障', 72, 'object'), ('鼓风机', 76, 'subject'), ('有杂物', 80, 'object')]
	```

- 接着我们可以着手关系抽取了，在re_mid_data下新建一个rels.txt，里面输入该数据集的关系，这里是：

	```python
	部件故障
	性能故障
	检测工具
	组成
	未知
	```

	我们新增了一项未知项，以解决主体和客体之间不存在关系的情况。在pytorch_triple_extraction/re_process.py里面修改路径为该数据集的路径，然后运行```python re_process.py```即可获得re_mid_data下的其它文件，看看里面数据：

	```python
	部件故障	故障现象：该车最多只能跑到120KM/H,再踩#油门#就$不起作用$了;	24	27	29	34
	```

	第一项为关系类别，第二项为文本，注意，我们在主体左右加入#标识，在客体左右加入$标识，最后四项是主客体的起始和结束位置。**注意：这里索引都已经提前+1，因为bert文本前面会加一个[CLS]** 。
	```python
	cd bert_re
	"""以下舍弃了，数据量太大会有问题
	在preprocess.py里面，修改data_name = "dgre"，并新增一个if-else分支，
	if data_name == "dgre":
	    args.max_seq_len = 512
	    args.data_dir = '../data/dgre/'
	    re_mid_data_path = '../data/dgre/re_mid_data'
	    
	最后运行python preprocess.py即可获得re_final_data下的文件。
	"""
	
	在main.py里面，我们需要做的是修改最后预测的那部分，这里要根据自己数据修改：
	text = '62号汽车故障报告综合情况:故障现象:加速后，丢开油门，#发动机#$熄火$。'
	ids = [29,	33,	34,	37]
	print('预测标签：', trainer.predict(tokenizer, text, id2label, args, ids))
	print('真实标签：', '部件故障')
	
	最后运行以下指令进行训练、验证、测试和预测：
	python main.py \
	--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
	--data_dir="../data/dgre/" \
	--log_dir="./logs/" \
	--output_dir="./checkpoints/" \
	--num_tags=5 \  # 根据rels.txt里面数目而定
	--seed=123 \
	--gpu_ids="0" \
	--max_seq_len=512 \
	--lr=3e-5 \
	--other_lr=3e-4 \
	--train_batch_size=8 \
	--train_epochs=1 \
	--eval_batch_size=8 \
	--dropout_prob=0.3 
	
	【test】 loss：7.225020 accuracy：0.9893 micro_f1：0.9893 macro_f1：0.8997
	预测标签： ['部件故障']
	真实标签： 部件故障
	```

	**注意：在测试时如果里面不含某类关系的数据，会报错：ValueError: Number of classes, 4, does not match size of target_names, 5. Try specifying the labels parameter**，所以在该数据上要把测试报告那部分代码注释掉。

- 实体和关系都训练完，我们会得到bert_bilstm_crf_ner/checkpoints/bert_crf/model.pt和bert_re/checkpoints/best.pt。在pytorch_triple_extraction/get_result.py里面进行融合预测，需要修改：
	```python
	model_name = 'bert_crf'  # 这些参数和之前的对应
	ner_args.use_lstm = 'False'
	ner_args.use_crf = 'True'
	ner_args.num_tags = 5
	ner_args.max_seq_len = 512
	
	re_args.num_tags = 5
	re_args.max_seq_len = 512
	    
	并最后修改预测的文本：
	 raw_texts = [
	    '故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失',
	    '1045号汽车故障报告故障现象打开点火开关，操作左前电动座椅开关，座椅6个方向均不动作故障原因六向电动座椅线束磨破搭铁修复方法包扎磨破线束，从新固定。',
	]
	    
	结果：
	('转向', 5, 'subject')
	('转向机', 19, 'subject')
	('转向轴', 23, 'subject')
	('缺油', 27, 'object')
	[('转向', 5, 7), ('转向机', 19, 22), ('转向轴', 23, 26)]
	[('缺油', 27, 29)]
	==========================
	故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失
	主体： ('转向', 5, 7)
	客体： ('缺油', 27, 29)
	关系： 部件故障
	==========================
	故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失
	主体： ('转向机', 19, 22)
	客体： ('缺油', 27, 29)
	关系： 部件故障
	==========================
	故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失
	主体： ('转向轴', 23, 26)
	客体： ('缺油', 27, 29)
	关系： 部件故障
	('座椅', 33, 'subject')
	('不动作', 40, 'object')
	('六向电动座椅线束', 47, 'subject')
	('磨破', 55, 'object')
	[('座椅', 33, 35), ('六向电动座椅线束', 47, 55)]
	[('不动作', 40, 43), ('磨破', 55, 57)]
	==========================
	1045号汽车故障报告故障现象打开点火开关，操作左前电动座椅开关，座椅6个方向均不动作故障原因六向电动座椅线束磨破搭铁修复方法包扎磨破线束，从新固定。
	主体： ('座椅', 33, 35)
	客体： ('不动作', 40, 43)
	关系： 部件故障
	==========================
	1045号汽车故障报告故障现象打开点火开关，操作左前电动座椅开关，座椅6个方向均不动作故障原因六向电动座椅线束磨破搭铁修复方法包扎磨破线束，从新固定。
	主体： ('座椅', 33, 35)
	客体： ('磨破', 55, 57)
	关系： 部件故障
	==========================
	1045号汽车故障报告故障现象打开点火开关，操作左前电动座椅开关，座椅6个方向均不动作故障原因六向电动座椅线束磨破搭铁修复方法包扎磨破线束，从新固定。
	主体： ('六向电动座椅线束', 47, 55)
	客体： ('不动作', 40, 43)
	关系： 部件故障
	==========================
	1045号汽车故障报告故障现象打开点火开关，操作左前电动座椅开关，座椅6个方向均不动作故障原因六向电动座椅线束磨破搭铁修复方法包扎磨破线束，从新固定。
	主体： ('六向电动座椅线束', 47, 55)
	客体： ('磨破', 55, 57)
	关系： 部件故障
	```

后话：之前的duie关系抽取没有考虑到数据单独建一个文件夹，可酌情修改，主要是一些路径问题。而对于上述数据集而言，也可以增加一些约束，比如：**客体要约束在主体之后，而不能在主体之前**。延伸到方面级的情感分析也是一样的。

## 总结

如果你想完成上面的实验，以下是步骤：

```python
下载chinese-bert-wwm-ext到pytorch_triple_extraction/model_hub下
==========================
cd pytorch_triple_extraction/data/dgre/raw_data
python process.py
==========================
cd pytorch_triple_extraction/bert_bilstm_crf_ner/
python preprocess.py
==========================
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="../data/dgre/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=5 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=512 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=8 \
--train_epochs=5 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm="False" \
--use_crf="True" \
--dropout_prob=0.3 \
--dropout=0.3 
==========================
cd pytorch_triple_extraction
python re_process.py
==========================
cd pytorch_triple_extraction/bert_re
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="../data/dgre/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=5 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=512 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=8 \
--train_epochs=1 \
--eval_batch_size=8 \
--dropout_prob=0.3 
==========================
cd pytorch_triple_extraction/
python get_result.py
```

****

# 最初的介绍

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
### 温馨提示
不要在pycharm里面直接运行，在命令行使用带参数运行，即main.py后面的一连串东东。

# 命名实体识别
在bert_bilstm_crf_ner文件夹下的main.py是主运行程序，进入到bert_bilstm_crf_ner文件夹下，可用以下命令运行训练测试和预测：<br>
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
### 温馨提示
- 由于数据量太大，在关系抽取main.py里面限制了在4000步保存模型并停止，可以酌情修改。
- 在main.py里面的训练、验证、测试和预测代码根据需要进行注释或打开。
- pytorch_triple_extraction/data/mid_re_data/rels.txt里面最后有一个空的，因此num_tags=实际标签数+1，这里就不进行改动了，有需要的话可以删除掉最后的空标签，那么num_tags就是实际标签数。（不使用我已经训练好的模型）。
- ids里面对应得索引已经+1了，因为前面有个CLS。

在bert_re文件夹下的main.py是主运行程序，进入到bert_re文件夹下，可用以下命令运行训练测试和预测：<br>
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
```python
python get_result.py
```
```python
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
