import torch
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer
from transformers.integrations import TensorBoardCallback
from datasets import Dataset

from utils import *
from wrapper import *

import argparse
import os
import logging
from time import time

parser = argparse.ArgumentParser(description='Model and Training Config')

## model parameters
parser.add_argument('--model_name', type=str, help='Huggingface model code for the Bert model', required=True)

## dataset parameters
parser.add_argument('--data_dir', type=str, help='Path to directory storing train.csv and test.csv files.', required=True)
parser.add_argument('--csv_sep', type=str, default='\t', help='delimiter used when reading csv files.')
parser.add_argument('--maxlength', type=int, default=128, help='Maximum length of input text.')
parser.add_argument('--num_training_examples', type=int, default=-1, help='Number of examples used for training. Pass -1 if using all examples.')
parser.add_argument('--split_words', default=False, action='store_true', help='Whether to perform jieba word splits. Default to False.')
parser.add_argument('--remove_username', default=False, action='store_true', help='Whether remove ids in texts. Default to True.')
parser.add_argument('--remove_punctuation', default=False, action='store_true', help='Whether remove punctuations in texts. Default to True.')
parser.add_argument('--to_simplified', default=False, action='store_true', help='Whether to convert all texts to simplified Chinese. Default to True.')
parser.add_argument('--emoji_to_text', default=False, action='store_true', help='Whether to translate emojis to texts. Default to True.')

## training parameters
parser.add_argument('--output_model_dir', type=str, help='Directory to store finetuned models', required=True)
parser.add_argument('--num_epochs', type=int, default=10, help='epoch', required=False)
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate ', required=False)
parser.add_argument('--batch_size', type=int, default=8, help='batch size', required=False)
parser.add_argument('--adversarial_training_param', type=int, default=0, help='Adversarial training parameter. If passed 0 then switched off adversarial traiing.')
parser.add_argument('--alpha', type=float, default=1, help='alpha parameter for focal loss. Change the weights of positive examples.')
parser.add_argument('--gamma', type=float, default=0, help='gamma parameter for focal loss. Change how strict the positive labelling is.')

## output parameters
# parser.add_argument('--perform_testing', help='Whether to use the trained model on a test set', default=False, action='store_true')
# parser.add_argument('--do_pred_on_dev_set', default=False, action='store_true')
# parser.add_argument('--pred_output_dir', type=str, help='Directory to store final test set predictions', default=None)
# parser.add_argument('--weighted_averaging', default=True, help='Whether to weight the logits of the model based on their dev set accuracy when creating ensemble.')
args = parser.parse_args()

# open('.log', 'w').close()    # clear logging file
if not os.path.exists(args.output_model_dir):
    os.makedirs(args.output_model_dir)
logging.basicConfig(filename=os.path.join(args.output_model_dir, str(time()).split('.')[1]+'.log'), encoding='utf-8', level=logging.INFO)

# Loading full datasets
data_files = [os.path.join(*(args.data_dir.split('/')), 'train.csv')]

logging.info(f'Reading training data from {data_files[0]}...')
train_df = pd.read_csv(data_files[0], sep=args.csv_sep) 
train_df = train_df[train_df.label == 0].reset_index().drop(columns=['id'])    # only get the error-free sentences

if args.num_training_examples == -1:
    logging.info('Using full training set.')
    train_df_use = train_df.drop_duplicates()
else:
    logging.info(f'Using randomly selected {args.num_training_examples} training examples.')
    train_df = train_df.drop_duplicates()
    train_df_use = train_df.iloc[np.random.choice(np.arange(len(train_df)), size=args.num_training_examples, replace=False)]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### ----------------------------- PREPARE DATASETS -----------------------------
logging.info(f'Constructing training dataset object')
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, return_tensors='pt', 
    )
def tokenization(example):
    return tokenizer(example["text"])

dev_set_idx = np.random.choice(np.arange(len(train_df_use)), size=len(train_df_use)//10, replace=False)
dev_set_df = train_df_use.iloc[dev_set_idx]
full_train_set_idx = np.array(list(set(np.arange(len(train_df_use))) - set(dev_set_idx)))
train_ds = Dataset.from_pandas(df=train_df_use.iloc[full_train_set_idx][['text']]).map(tokenization, batched=True)
dev_ds = Dataset.from_pandas(df=dev_set_df[['text']]).map(tokenization, batched=True)

### ----------------------------- TRAINING -----------------------------
# Set up training 
logging.info(f'Training Starts')

model = AutoModelForMaskedLM.from_pretrained(args.model_name)

if 'cuda' in DEVICE.type:
    model.cuda()

# Initialise tensorboard
tb_writer = SummaryWriter(os.path.join(args.output_model_dir, 'runs'))

# Define training arguments
logging.info('Defining TrainingArguments.')
arguments = TrainingArguments(
    output_dir=args.output_model_dir, 
    per_device_train_batch_size=args.batch_size, 
    per_device_eval_batch_size=args.batch_size, 
    num_train_epochs=args.num_epochs,
    logging_steps=100, 
    evaluation_strategy="epoch", # run validation at the end of each epoch
    # evaluation_strategy="steps",
    # eval_steps=500, 
    save_strategy="epoch",  # save checkpoint at each epoch
    # save_strategy="steps",
    # save_steps=500, 
    learning_rate=args.lr, 
    load_best_model_at_end=True, 
    report_to='tensorboard',
    push_to_hub=False, 
)

trainer = Trainer(
    model=model,
    args=arguments, 
    train_dataset=train_ds, 
    eval_dataset=dev_ds, 
    data_collator=data_collator,
)

trainer.add_callback(TensorBoardCallback(tb_writer=tb_writer))

# Train the model
trainer.train()

del model
torch.cuda.empty_cache()
