#!/usr/bin/env bash
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
--dropout_prob=0.3 \