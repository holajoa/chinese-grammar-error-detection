from urllib.parse import DefragResult
import torch
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback

from utils import *
from dataset import *
from preprocess import *
from wrapper import *
from models import AutoModelWithNER, AutoModelWithClassificationHead, BertWithCRFHead

import argparse
import os
import logging
from time import time

# open('.log', 'w').close()    # clear logging file
logging.basicConfig(filename=os.path.join('logs', str(time()).split('.')[1]+'.log'), encoding='utf-8', level=logging.INFO)
parser = argparse.ArgumentParser(description='Model and Training Config')

## model parameters
parser.add_argument('--model_name', type=str, help='Huggingface model code for the Bert model', required=True)
parser.add_argument('--num_labels', type=int, default=2, help='Number of classes in the dataset', required=True)
parser.add_argument('--ner_model_name', type=str, help='Finetuned model for named entities recognition boosting')
parser.add_argument('--single_layer_cls_head', default=False, action='store_true')
parser.add_argument('--add_up_hiddens', default=False, action='store_true')
parser.add_argument('--token_level_model', default=False, action='store_true')

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
parser.add_argument('--kfolds', type=int, default=5, help='k-fold k', required=False)
parser.add_argument('--folds', type=str, help='Directory to txt file storing the fold indices used for training.')
parser.add_argument('--fold_size', type=int, help='Number of examples in a fold', required=False)

## training parameters
parser.add_argument('--output_model_dir', type=str, help='Directory to store finetuned models', required=True)
parser.add_argument('--seed', type=int)
parser.add_argument('--n_fold_used', type=int, help='Number of folds of data used for training each model.')
parser.add_argument('--num_ensemble_models', type=int, help='Number of trained models.')
parser.add_argument('--epoch', type=int, default=10, help='epoch', required=False)
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate ', required=False)
parser.add_argument('--batch_size', type=int, default=8, help='batch size', required=False)
parser.add_argument('--best_by_f1', default=False, action='store_true', help='Call back to best model by F1 score.')
parser.add_argument('--adversarial_training_param', type=int, default=0, help='Adversarial training parameter. If passed 0 then switched off adversarial traiing.')
parser.add_argument('--alpha', type=float, default=1, help='alpha parameter for focal loss. Change the weights of positive examples.')
parser.add_argument('--gamma', type=float, default=0, help='gamma parameter for focal loss. Change how strict the positive labelling is.')
parser.add_argument('--calibration_temperature', type=float, default=1)
parser.add_argument('--local_loss_param', type=float, default=1e-2, help='Hyperparameter for token-level local loss.')
parser.add_argument('--early_stopping_patience', type=int, default=4)
parser.add_argument('--resume_fold_idx', type=int, help='On which fold to resume training.')
parser.add_argument('--checkpoint', type=str, help='previous model checkpoint.')
parser.add_argument('--from_another_run', default=False, action='store_true')

## output parameters
parser.add_argument('--perform_testing', help='Whether to use the trained model on a test set', default=False, action='store_true')
parser.add_argument('--pred_output_dir', type=str, help='Directory to store final test set predictions', default=None)
# parser.add_argument('--weighted_averaging', default=True, help='Whether to weight the logits of the model based on their dev set accuracy when creating ensemble.')
args = parser.parse_args()



# Loading full datasets
data_files = [os.path.join(*(args.data_dir.split('/')), 'train.csv')]

if args.perform_testing:
    data_files.append(os.path.join(*(args.data_dir.split('/')), 'test.csv'))
    assert args.pred_output_dir, 'Must procvide an output path for predictions on the test set.'

logging.info(f'Reading training data from {data_files[0]}...')
train_df = pd.read_csv(data_files[0], sep=args.csv_sep).set_index('id')

if args.num_training_examples == -1:
    logging.info('Using full training set.')
    train_df_use = train_df
else:
    logging.info(f'Using randomly selected {args.num_training_examples} training examples.')
    train_df_use = train_df.iloc[np.random.choice(np.arange(len(train_df)), size=args.num_training_examples, replace=False)]

if args.perform_testing:
    print(data_files[1])
    logging.info(f'Reading test data from {data_files[1]}...')
    test_df = pd.read_csv(data_files[1], sep=args.csv_sep).set_index('id')
else: 
    logging.info('No testing stage.')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DEV_DATASET_CONFIGS = {
    'model_name':args.model_name,
    'aux_model_name':args.ner_model_name, 
    'maxlength':args.maxlength,
    'train_val_split':-1,
    'test':False, 
    'split_words':args.split_words,
    'remove_username':args.remove_username,
    'remove_punctuation':args.remove_punctuation, 
    'to_simplified':args.to_simplified, 
    'emoji_to_text':args.emoji_to_text,
    'device':DEVICE, 
    'split_words':True, 
    'cut_all':False, 
}

entities_file = os.path.join(
    "D:\Apps\Anaconda3\envs\general-torch\Lib\site-packages", 
    "nlpcda\data\entities.txt", 
)

DATA_AUG_CONFIGS = {
    'random_entity':{
        'base_file':entities_file, 
        'create_num':2, 
        'change_rate':0.75, 
        'seed':1024, 
        'prop':0.3,  
    }, 
    'random_delete_char':{
        'create_num':2, 
        'change_rate':0.05, 
        'seed':1024, 
        'prop':0.1, 
    }, 
    'random_swap':{
        'create_num':2, 
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.2, 
    }, 
    'random_swap_order':{
        'create_num':2,
        'char_gram':5,  
        'change_rate':0.05, 
        'seed':1024, 
        'prop':0.5, 
    }
}


### ----------------------------- PREPARE DATASETS -----------------------------
save_generated_datasets_dir = os.path.join(args.output_model_dir, 'data')

# If provided, load dev set and training fold indices from file
if args.folds:
    logging.info(f'Loading dev set indices from file {args.folds}.')
    with open(args.folds, 'r') as f:
        lines = f.readlines()
        folds = [np.array(list(map(int, line.rstrip().split(' ')))) for line in lines]
        dev_set_idx, folds = folds[0], folds[1:]
    assert sum(map(len, folds)) == len(train_df_use), \
        'Provided dev set and training fold indices does not match with number of training examples.'

# Otherwise, randomly cast aside 1/10 of the original training data as dev set
dev_set_idx = np.random.choice(np.arange(len(train_df_use)), size=len(train_df_use)//10, replace=False)

# Get the dataframes from indices and save to file
dev_set_df = train_df.iloc[dev_set_idx]
if not os.path.exists(save_generated_datasets_dir):
    os.makedirs(save_generated_datasets_dir)
dev_set_df.to_csv(os.path.join(save_generated_datasets_dir, 'dev.csv'), sep='\t', index=False)

# Use remaining to construct folds during training
full_train_set_idx = np.array(list(set(np.arange(len(train_df_use))) - set(dev_set_idx)))
# train_df_use = train_df_use.iloc[full_train_set_idx]

# Construct folds - use provided if exists, otherwise generate folds on the fly
if args.folds:
    assert folds
    k = len(folds)
    logging.info(f'Loading {k} training folds from file {args.folds}.')
else:
    k = args.kfolds
    minority_idx = np.argwhere((train_df_use.label == 0).values).flatten()
    folds = easy_ensenble_generate_kfolds(full_train_set_idx, k, minority_idx)

# Write folds indices to file
with open(os.path.join(save_generated_datasets_dir, 'folds.txt'), 'a') as fp:
    fp.write(' '.join(list(map(str, dev_set_idx))) + '\n')
    for item in folds:
        fp.write(' '.join(list(map(str, item))) + '\n')
for fi in range(k):
    train_df_single_fold = train_df_use.iloc[folds[fi]]
    train_df_single_fold.to_csv(os.path.join(save_generated_datasets_dir, f'train_{fi}.csv'), sep='\t', index=False)
    # construct dataset objects for each fold


# Set up training 
n_fold_use = args.n_fold_used if args.n_fold_used else k
logging.info(f'When training each individual model, {n_fold_use} out of {k} folds of data will be used.')
n_models = args.num_ensemble_models

# Set up dev set dataset object
logging.info('Setting up dev set.')
dev_set = DatasetWithAuxiliaryEmbeddings(df=dev_set_df, **TRAIN_DEV_DATASET_CONFIGS)
dev_set.prepare_dataset()

for i in range(n_models):
    logging.info('=' * 50 + f'Training stage {i+1}/{n_models}' + '=' * 50)
    # Set up training set
    fold_idx = np.random.choice(range(k), size=n_fold_use, replace=False)
    train_idx_single_model = folds[fold_idx].flatten()
    train_df_single_model = train_df_use.iloc[train_idx_single_model]
    logging.info(f'Constructing {i+1} training dataset object')
    train = DatasetWithAuxiliaryEmbeddings(df=train_df_single_model, **TRAIN_DEV_DATASET_CONFIGS)
    train.prepare_dataset()