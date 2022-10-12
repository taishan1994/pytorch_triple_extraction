import os
import logging
import numpy as np
import torch
# 之前是自定义评价指标
try:
  from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
  import config
  import dataset
  from preprocess import BertFeature
  import bert_ner_model
except Exception as e:
  from .utils import commonUtils, metricsUtils, decodeUtils, trainUtils
  from . import config
  from . import dataset
  from .preprocess import BertFeature
  from . import bert_ner_model
# 现在我们使用seqeval库里面的
from seqeval.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

# 要显示传入BertFeature

from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

args = config.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)



class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        model = bert_ner_model.BertNerModel(args)
        self.model, self.device = trainUtils.load_model_and_parallel(model, args.gpu_ids)
        if self.train_loader:
          self.t_total = len(self.train_loader) * args.train_epochs
          self.optimizer, self.scheduler = trainUtils.build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 1 #每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    if key != 'texts':
                        batch_data[key] = batch_data[key].to(self.device)
                loss, logits = self.model(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'], batch_data['labels'])

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                # loss.backward(loss.clone().detach())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                global_step += 1
                """这里验证耗时有点长，我们最后直接保存模型就好
                if global_step % eval_steps == 0:
                    dev_loss, accuracy, precision, recall, f1 = self.dev()
                    logger.info('[eval] loss:{:.4f} accuracy:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, accuracy, precision, recall, f1))
                    if f1 > best_f1:
                        trainUtils.save_model(args, self.model, model_name, global_step)
                        f1 = f1_score
                """
        trainUtils.save_model(args, self.model, model_name)        

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            batch_output_all = []
            batch_true_all = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_loss, dev_logits = self.model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'], dev_batch_data['labels'])
                tot_dev_loss += dev_loss.item()
                if self.args.use_crf == 'True':
                    batch_output = dev_logits
                else:
                    batch_output = dev_logits.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=2)
                if len(batch_output_all) == 0:
                    batch_output_all = batch_output
                    # 获取真实的长度标签
                    tmp_labels = dev_batch_data['labels'].detach().cpu().numpy()
                    tmp_masks = dev_batch_data['attention_masks'].detach().cpu().numpy()
                    # print(tmp_labels.shape)
                    # print(tmp_masks.shape)
                    batch_output_all = [list(map(lambda x:self.idx2tag[x], i)) for i in batch_output_all]
                    batch_true_all = [list(tmp_labels[i][tmp_masks[i]==1]) for i in range(tmp_labels.shape[0])]
                    batch_true_all = [list(map(lambda x:self.idx2tag[x], i)) for i in batch_true_all]
                    # print(batch_output_all[1])
                    # print(batch_true_all[1])
                else:
                    batch_output = [list(map(lambda x:self.idx2tag[x], i)) for i in batch_output]
                    batch_output_all = np.append(batch_output_all, batch_output, axis=0)
                    tmp_labels = dev_batch_data['labels'].detach().cpu().numpy()
                    tmp_masks = dev_batch_data['attention_masks'].detach().cpu().numpy()
                    tmp_batch_true_all = [list(tmp_labels[i][tmp_masks[i]==1]) for i in range(tmp_labels.shape[0])]
                    tmp_batch_true_all = [list(map(lambda x:self.idx2tag[x], i)) for i in tmp_batch_true_all]
                    batch_true_all = np.append(batch_true_all, tmp_batch_true_all, axis=0)
            accuracy = accuracy_score(batch_true_all, batch_output_all)
            precision = precision_score(batch_true_all, batch_output_all) 
            recall = recall_score(batch_true_all, batch_output_all)
            f1 = f1_score(batch_true_all, batch_output_all)
            return tot_dev_loss, accuracy, precision, recall, f1

    def test(self, model_path):
        model = bert_ner_model.BertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        pred_label = []
        true_label = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(self.test_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                _, logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'],dev_batch_data['labels'])
                if self.args.use_crf == 'True':
                    batch_output = logits
                else:
                    batch_output = logits.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=2)
                if len(pred_label) == 0:
                    tmp_labels = dev_batch_data['labels'].detach().cpu().numpy()
                    tmp_masks = dev_batch_data['attention_masks'].detach().cpu().numpy()
                    pred_label = [list(map(lambda x:self.idx2tag[x], i)) for i in batch_output]
                    # true_label = dev_batch_data['labels'].detach().cpu().numpy().tolist()
                    true_label = [list(tmp_labels[i][tmp_masks[i]==1]) for i in range(tmp_labels.shape[0])]
                    true_label = [list(map(lambda x:self.idx2tag[x], i)) for i in true_label]
                    print(pred_label)
                    print(true_label)
                else:
                    # pred_label = np.append(pred_label, batch_output, axis=0)
                    # true_label = np.append(pred_label, dev_batch_data['labels'].detach().cpu().numpy().tolist(), axis=0)
                    batch_output = [list(map(lambda x:self.idx2tag[x], i)) for i in batch_output]
                    pred_label = np.append(pred_label, batch_output, axis=0)
                    tmp_labels = dev_batch_data['labels'].detach().cpu().numpy()
                    tmp_masks = dev_batch_data['attention_masks'].detach().cpu().numpy()
                    tmp_batch_true_all = [list(tmp_labels[i][tmp_masks[i]==1]) for i in range(tmp_labels.shape[0])]
                    tmp_batch_true_all = [list(map(lambda x:self.idx2tag[x], i)) for i in tmp_batch_true_all]
                    true_label = np.append(true_label, tmp_batch_true_all, axis=0)
            logger.info(classification_report(true_label, pred_label))

    def predict(self, raw_text, model_path):
        model = bert_ner_model.BertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            # tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
            tokens = [i for i in raw_text]
            encode_dict = tokenizer.encode_plus(text=tokens,
                                    max_length=self.args.max_seq_len,
                                    padding='max_length',
                                    truncation='longest_first',
                                    is_pretokenized=True,
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            # tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'],dtype=np.uint8)).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
            logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
            if self.args.use_crf != "True":
              logits = logits.detach().cpu().numpy()
              logits = np.argmax(output, axis=2)
            pred_label = [list(map(lambda x:self.idx2tag[x], i)) for i in logits]
            assert len(pred_label[0]) == len(tokens)+2
            pred_entities = decodeUtils.get_entities(pred_label[0][1:1+len(tokens)], "".join(tokens))
            logger.info(pred_entities)
            return pred_entities


if __name__ == '__main__':
    data_name = 'dgre'
    # args.train_epochs = 1
    # args.train_batch_size = 32
    # args.max_seq_len = 300
    model_name = ''
    if args.use_lstm == 'True' and args.use_crf == 'False':
        model_name = 'bert_bilstm'
    if args.use_lstm == 'True' and args.use_crf == 'True':
        model_name = 'bert_bilstm_crf'
    if args.use_lstm == 'False' and args.use_crf == 'True':
        model_name = 'bert_crf'
    if args.use_lstm == 'False' and args.use_crf == 'False':
        model_name = 'bert'
    commonUtils.set_logger(os.path.join(args.log_dir, '{}.log'.format(model_name)))


    data_path = os.path.join(args.data_dir, 'ner_final_data')
    mid_data_path = os.path.join(args.data_dir, 'mid_data')
    # 真实标签
    ent_labels_path = mid_data_path + '/ent_labels.txt'
    # 序列标注标签B I O
    ner_labels_path = mid_data_path + '/ner_labels.txt'
    with open(ent_labels_path, 'r') as fp:
        ent_labels = fp.read().strip().split('\n')
    entlabel2id = {}
    id2entlabel = {}
    for i,j in enumerate(ent_labels):
      entlabel2id[j] = i
      id2entlabel[i] = j
    nerlabel2id = {}
    id2nerlabel = {}
    with open(ner_labels_path,'r') as fp:
        ner_labels = fp.read().strip().split('\n')
    for i,j in enumerate(ner_labels):
      nerlabel2id[j] = i
      id2nerlabel[i] = j
    logger.info(id2nerlabel)
    args.num_tags = len(ner_labels)
    logger.info(args)

    train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
    train_dataset = dataset.NerDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)
    dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
    dev_dataset = dataset.NerDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)
    # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
    # test_dataset = dataset.NerDataset(test_features)
    # test_loader = DataLoader(dataset=test_dataset,
    #                         batch_size=args.eval_batch_size,
    #                         num_workers=2)
    bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2nerlabel)
    bertForNer.train()

    model_path = './checkpoints/{}/model.pt'.format(model_name)
    bertForNer.test(model_path)

    if data_name == "dgre":
        raw_text = "211号汽车故障报告综合情况:故障现象:开暖风鼓风机运转时有异常响声。故障原因简要分析:该故障是鼓风机运转时有异响由此可以判断可能原因：1鼓风机故障 2鼓风机内有杂物"
    elif data_name == "duie":
        raw_text = "《单身》是Outsider演唱的歌曲，收录于专辑《2辑Maestro》。描写一个人单身的感觉，单身即是痛苦的也是幸福的，在于人们如何去看待s"

    logger.info(raw_text)
    bertForNer.predict(raw_text, model_path)
