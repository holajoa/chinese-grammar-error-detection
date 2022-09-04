from urllib.parse import DefragResult
import torch
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback

from utils import *
from dataset import *
from preprocess import *
from wrapper import *
from models import AutoModelWithOOBModel

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
parser.add_argument('--oob_model_name', type=str, help='Finetuned model for zero-shot extra information')
parser.add_argument('--add_hidden_states', default=False, action='store_true')

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
parser.add_argument('--minor_major_ratio', type=float, default=1.)

## training parameters
parser.add_argument('--output_model_dir', type=str, help='Directory to store finetuned models', required=True)
parser.add_argument('--seed', type=int)
parser.add_argument('--n_fold_used', type=int, help='Number of folds of data used for training each model.')
parser.add_argument('--num_ensemble_models', type=int, help='Number of trained models.')
parser.add_argument('--num_epochs', type=int, default=10, help='epoch', required=False)
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate ', required=False)
parser.add_argument('--batch_size', type=int, default=8, help='batch size', required=False)
parser.add_argument('--best_by_f1', default=False, action='store_true', help='Call back to best model by F1 score.')
parser.add_argument('--adversarial_training_param', type=int, default=0, help='Adversarial training parameter. If passed 0 then switched off adversarial traiing.')
parser.add_argument('--alpha', type=float, default=1, help='alpha parameter for focal loss. Change the weights of positive examples.')
parser.add_argument('--gamma', type=float, default=0, help='gamma parameter for focal loss. Change how strict the positive labelling is.')
parser.add_argument('--early_stopping_patience', type=int, default=4)
parser.add_argument('--resume_model_idx', type=int, help='On which fold to resume training.')

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
    train_df_use = train_df.drop_duplicates()
else:
    logging.info(f'Using randomly selected {args.num_training_examples} training examples.')
    train_df = train_df.drop_duplicates()
    train_df_use = train_df.iloc[np.random.choice(np.arange(len(train_df)), size=args.num_training_examples, replace=False)]

if args.perform_testing:
    print(data_files[1])
    logging.info(f'Reading test data from {data_files[1]}...')
    test_df = pd.read_csv(data_files[1], sep=args.csv_sep).set_index('id')
else: 
    logging.info('No testing stage.')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlpcda_data_path = 'D:/Apps/Anaconda3/envs/general-torch/Lib/site-packages/nlpcda/data'
entities_file = os.path.join(nlpcda_data_path, "entities.txt")
logic_pairs_file = os.path.join(nlpcda_data_path, 'logic_pairs.txt')

DATA_AUG_CONFIGS = {
    'random_entity':{
        'base_file':entities_file, 
        'create_num':2, 
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.25,  
    }, 
    'random_delete_char':{
        'create_num':2, 
        'change_rate':0.1, 
        'seed':1024, 
        'prop':0.25, 
    }, 
    'random_swap':{
        'create_num':2, 
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.25, 
    }, 
    'random_swap_order':{
        'create_num':2,
        'char_gram':5,  
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.25, 
    }, 
    'random_add_similar':{
        'create_num':2,
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.25, 
    }, 
    'random_swap_logic_words':{
        'base_file':logic_pairs_file, 
        'create_num':2,
        'change_rate':0.5, 
        'seed':1024, 
        'prop':0.5, 
    }, 
}

TRAIN_DATASET_CONFIGS = {
    'model_name':args.model_name,
    'aux_model_name':args.oob_model_name, 
    'maxlength':args.maxlength,
    'train_val_split':-1,
    'test':False, 
    'split_words':args.split_words,
    'remove_username':args.remove_username,
    'remove_punctuation':args.remove_punctuation, 
    'to_simplified':args.to_simplified, 
    'emoji_to_text':args.emoji_to_text,
    'device':DEVICE, 
    'split_words':False, 
    'cut_all':False, 
    'da_configs':DATA_AUG_CONFIGS, 
}

DEV_DATASET_CONFIGS = {
    'model_name':args.model_name,
    'aux_model_name':args.oob_model_name, 
    'maxlength':args.maxlength,
    'train_val_split':-1,
    'test':False, 
    'split_words':args.split_words,
    'remove_username':args.remove_username,
    'remove_punctuation':args.remove_punctuation, 
    'to_simplified':args.to_simplified, 
    'emoji_to_text':args.emoji_to_text,
    'device':DEVICE, 
    'split_words':False, 
    'cut_all':False, 
}

TEST_DATASET_CONFIGS = {
    'model_name':args.model_name,
    'aux_model_name':args.oob_model_name, 
    'maxlength':args.maxlength,
    'train_val_split':-1,
    'test':True, 
    'split_words':args.split_words,
    'remove_username':args.remove_username,
    'remove_punctuation':args.remove_punctuation, 
    'to_simplified':args.to_simplified, 
    'emoji_to_text':args.emoji_to_text,
    'device':DEVICE, 
    'split_words':False, 
    'cut_all':False, 
}


### ----------------------------- PREPARE DATASETS -----------------------------
save_generated_datasets_dir = os.path.join(args.output_model_dir, 'data')

# If provided, load dev set and training fold indices from file
if args.folds:
    logging.info(f'Loading dev set indices from file {args.folds}.')
    with open(args.folds, 'r') as f:
        lines = f.readlines()
        folds = [np.array(list(map(int, line.rstrip().split(' ')))) for line in lines]
        if sum(map(len, folds)) != len(train_df_use):
            logging.warning(
                f'Number of provided dev set and training fold indices ({sum(map(len, folds))}) ' 
                f'does not match with number of training examples ({len(train_df_use)}).'
            )
        dev_set_idx, folds = np.array(folds[0]), np.array(folds[1:])
    

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
    assert folds is not None
    k = len(folds)
    logging.info(f'Loading {k} training folds from file {args.folds}.')
else:
    k = args.kfolds
    minority_idx = np.argwhere((train_df_use.label == 0).values).flatten()
    folds = easy_ensenble_generate_kfolds(full_train_set_idx, k, minority_idx, minor_major_ratio=args.minor_major_ratio)

# Write folds indices to file
with open(os.path.join(save_generated_datasets_dir, 'folds.txt'), 'w+') as fp:
    fp.write(' '.join(list(map(str, dev_set_idx))) + '\n')
    for item in folds:
        fp.write(' '.join(list(map(str, item))) + '\n')
for fi in range(k):
    train_df_single_fold = train_df_use.iloc[folds[fi]]
    train_df_single_fold.to_csv(os.path.join(save_generated_datasets_dir, f'train_{fi}.csv'), sep='\t', index=False)
    # construct dataset objects for each fold

# Set up dev set dataset object
logging.info('Setting up dev set.')
dev = DatasetWithAuxiliaryEmbeddings(df=dev_set_df, **DEV_DATASET_CONFIGS)
dev.prepare_dataset()

# Set up test set dataset object
if args.perform_testing:
    logging.info('Setting up test dataset to run predictions on.')
    test = DatasetWithAuxiliaryEmbeddings(df=test_df, **TEST_DATASET_CONFIGS)
    test.prepare_dataset()


### ----------------------------- TRAINING -----------------------------
# Set up training 
n_fold_use = args.n_fold_used if args.n_fold_used else k
logging.info(f'When training each individual model, {n_fold_use} out of {k} folds of data will be used.')
n_models = args.num_ensemble_models

if args.perform_testing:
    test_set_logits = []


irange = range(args.resume_model_idx, n_models) if args.resume_model_idx else range(n_models)

for i in irange:
    logging.info('=' * 50 + f'Training stage {i+1}/{n_models}' + '=' * 50)
    # Set up training set
    np.random.seed(int(time()))
    fold_idx = np.random.choice(range(k), size=n_fold_use, replace=False)
    train_idx_single_model = folds[fold_idx].flatten()
    train_df_single_model = train_df_use.iloc[train_idx_single_model].drop_duplicates()
    logging.info(f'Constructing {i+1} training dataset object')
    train = DatasetWithAuxiliaryEmbeddings(df=train_df_single_model, **TRAIN_DATASET_CONFIGS)
    train.prepare_dataset()

    model = AutoModelWithOOBModel(
        model=args.model_name, 
        oob_model=args.oob_model_name, 
        n_labels=args.num_labels, 
        concatenate=~(args.add_hidden_states), 
    )

    if 'cuda' in DEVICE.type:
        model.cuda()
    
    # Construct required directories
    single_model_output_dir = os.path.join(args.output_model_dir, 'model'+str(i))
    if not os.path.exists(single_model_output_dir):
        os.makedirs(single_model_output_dir)
    with open(os.path.join(single_model_output_dir, 'used_fold_indices.txt'), 'w+') as f:
        f.write(' '.join(list(map(str, fold_idx))))
    # Initialise tensorboard
    tb_writer = SummaryWriter(os.path.join(single_model_output_dir, 'runs'))

    # Define training arguments
    logging.info('Defining TrainingArguments.')
    seed = args.seed if args.seed else 42
    metric_for_best_model = 'F1' if args.best_by_f1 else 'loss'
    arguments = MyTrainingArguments(
        output_dir=single_model_output_dir, 
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
        metric_for_best_model=metric_for_best_model, 
        label_names=['labels'],   # need to specify this to pass the labels to the trainer
        epsilon=args.adversarial_training_param, 
        alpha=args.alpha, 
        gamma=args.gamma, 
        seed=seed,
        report_to='tensorboard',
        push_to_hub=False, 
    )

    trainer = ImbalancedTrainer(
        model=model,
        args=arguments, 
        train_dataset=train.dataset['train'], 
        eval_dataset=dev.dataset['train'],
        tokenizer=train.tokenizer, 
        compute_metrics=compute_metrics, 
    )
    
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience, 
        early_stopping_threshold=0.0, 
    ))  # apply early stopping - stop training immediately if the loss cease to decrease
    trainer.add_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Train the model
    trainer.train()

    if args.perform_testing:
        # Get logits on the test set
        test_set_hiddens = trainer.predict(test.dataset['train']).predictions[0]
        test_set_logits.append(test_set_hiddens)

    del model
    torch.cuda.empty_cache()

if args.perform_testing:
    logging.info('Aggregating predictions on test set ...')

    test_set_logits_agg = np.array(test_set_logits).mean(0)
    test_set_predictions = np.argmax(test_set_logits_agg, 1)
    test_set_result = pd.DataFrame(test_set_predictions, columns=['label'])
    test_set_result['id'] = range(1, 1+len(test_set_result))
    test_set_result = test_set_result[['id', 'label']]

    # Write results
    out_path = os.path.join(args.pred_output_dir, 'submission.csv')

    with open(out_path, 'w+') as f:
        test_set_result.to_csv(out_path, index=False, sep='\t')
