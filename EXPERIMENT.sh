#!/bin/bash
python3 HashBertMk15D.py --model_type distilbert --model_name_or_path distilbert-base-cased  --do_train --do_eval --data_dir .\Data --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir .\mk15D1 --overwrite_cache
python eval_script.py preds Data/Gold 

