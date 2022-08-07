from urllib.parse import DefragResult
import torch
import pandas as pd
import re
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback, TrainerCallback

from utils import *
from dataset import *
from preprocess import *
from wrapper import *
from models import BertWithNER

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
parser.add_argument('--adversarial_training_param', type=int, default=0, help='Adversarial training parameter. If passed 0 then switched off adversarial traiing.')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter for focal loss. Change the weights of positive examples.')
parser.add_argument('--gamma', type=float, default=0, help='gamma parameter for focal loss. Change how strict the labelling is.')
parser.add_argument('--batch_size', type=int, default=8, help='batch size', required=False)
parser.add_argument('--epoch', type=int, default=10, help='epoch', required=False)
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate ', required=False)
parser.add_argument('--kfolds', type=int, default=5, help='k-fold k', required=False)
parser.add_argument('--output_model_dir', type=str, help='Directory to store finetuned models', required=True)

## output parameters
parser.add_argument('--perform_testing', help='Whether to use the trained model on a test set', default=False, action='store_true')
parser.add_argument('--pred_output_dir', type=str, help='Directory to store final test set predictions', default=None)
# parser.add_argument('--weighted_averaging', default=True, help='Whether to weight the logits of the model based on their dev set accuracy when creating ensemble.')
args = parser.parse_args()



data_files = [os.path.join(args.data_dir, 'train.csv')]
if args.perform_testing:
    data_files.append(os.path.join(args.data_dir, 'test.csv'))
    assert args.pred_output_dir, 'Must procvide an output path for predictions on the test set.'

logging.info(f'Reading training data from {data_files[0]}...')
train_df = pd.read_csv(data_files[0], sep=args.csv_sep)
if args.num_training_examples == -1:
    logging.info('Using full training set.')
    train_df_use = train_df
else:
    logging.info(f'Using randomly selected {args.num_training_examples} training examples.')
    train_df_use = train_df.iloc[np.random.choice(np.arange(len(train_df)), size=args.num_training_examples, replace=False)]
    train_df_use = train_df_use.reset_index().drop(columns=['index'])

if args.perform_testing:
    logging.info(f'Reading test data from {data_files[1]}...')
    test_df = pd.read_csv(data_files[1], sep=args.csv_sep)
else: 
    logging.info('No testing stage.')

k = args.kfolds
train_val_split = 1 - 1/k

train_dataset_config = {
    'model_name':args.model_name,
    'maxlength':args.maxlength,
    'train_val_split':train_val_split,
    'test':False, 
    'split_words':args.split_words,
    'remove_username':args.remove_username,
    'remove_punctuation':args.remove_punctuation, 
    'to_simplified':args.to_simplified, 
    'emoji_to_text':args.emoji_to_text,
}

if args.perform_testing:
    test_dataset_config = {
        'model_name':args.model_name,
        'maxlength':args.maxlength,
        'train_val_split':-1,
        'test':True, 
        'split_words':args.split_words,
        'remove_username':args.remove_username,
        'remove_punctuation':args.remove_punctuation, 
        'to_simplified':args.to_simplified, 
        'emoji_to_text':args.emoji_to_text,
    }

    logging.info(f'Constructing test dataset object, with the following config:')
    logging.info(test_dataset_config)
    test = SimpleDataset(df=test_df, **test_dataset_config)
    test.tokenize()
    test.construct_dataset()


logging.info(f'Setting up {k}-fold CV.')
L = len(train_df_use)
folds = generate_folds(L, k)
val_accuracies = []
if args.perform_testing:
    logits = []


for i in range(k):
    val_idx = folds[i]
    logging.info(f'Training stage {i+1}/{k} ...')
    logging.info(f'Constructing {k}-fold training dataset object, with the following config:')
    logging.info(train_dataset_config)
    train = SimpleDataset(df=train_df_use, **train_dataset_config)
    train.tokenize()
    train.construct_dataset(val_idx=val_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.ner_model_name:
        model = BertWithNER(
            bert_model=args.model_name, 
            ner_model='uer/roberta-base-finetuned-cluener2020-chinese', 
            n_labels=2, 
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=args.num_labels,
        )   

    model.cuda()

    logging.info('Defining TrainingArguments.')
    # Define training arguments
    arguments = AdversarialTrainingArguments(
        output_dir=os.path.join(args.output_model_dir, 'fold'+str(i)),  # output directory
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        num_train_epochs=args.epoch,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",  # save checkpoint at each epoch
        learning_rate=args.lr, 
        load_best_model_at_end=True,
        label_names=['labels'],   # need to specify this to pass the labels to the trainer
        epsilon=args.adversarial_training_param, 
        alpha=args.alpha, 
        gamma=args.gamma, 
    )

    trainer = AdversarialTrainer(
        model=model, 
        args=arguments, 
        train_dataset=train.dataset['train'], 
        eval_dataset=train.dataset['val'],  
        tokenizer=train.tokenizer, 
        compute_metrics=compute_metrics
    )

    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=4, 
        early_stopping_threshold=0.0, 
    ))  # apply early stopping - stop training immediately if the loss cease to decrease

    # Train the model
    trainer.train()

    # Get soft labels of dev set 
    # dev_labels = np.argmax(trainer.predict(train.dataset['val']).predictions, axis=1)
    # soft_labels.append(dev_labels)
    
    # Get pred accuracy on the dev set 
    val_pred = np.argmax(trainer.predict(train.dataset['val']).predictions, 1)
    val_accuracy = (val_pred == train.dataset['val']['labels'].numpy()).mean()
    val_accuracies.append(val_accuracy)

    # Print the predictions
    # logging.info("Full model trained...")
    # logging.info(f"True labels:      {train.dataset['val']['input_ids']}")
    # logging.info(f'Predicted labels: {val_pred}')

    if args.perform_testing:
        # Get logits on the test set
        hiddens = trainer.predict(test.dataset['train']).predictions
        logits.append(hiddens)

    del model
    torch.cuda.empty_cache()


if args.perform_testing:
    logging.info('Doing predictions on test set ...')
    # Ensemble by logits
    # predictions = averaging(logits, val_accuracies, weighted=args.weighted_averaging)
    hiddens = np.array(logits).mean(0)
    predictions = np.argmax(hiddens, 1)
    result = pd.DataFrame(predictions, columns=['label'])

    # Write results
    out_path = os.path.join(args.pred_output_dir, 'submission.csv')

    with open(out_path, 'a+') as f:
        result.to_csv(out_path, index=False)
