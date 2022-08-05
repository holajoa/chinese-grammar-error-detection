import torch
import pandas as pd
import numpy as np
from transformers import pipeline

from datasets import *
from utils import *
from wrapper import *


torch.cuda.is_available()

def get_ner_score(df, model, tokenizer, device):
    ner = pipeline('ner', model=model, tokenizer=tokenizer, device=device)
    train_text = df if isinstance(df, pd.Series) else df['text']
    text_dataset = Dataset.from_pandas(train_text.to_frame())
    outputs = ner(text_dataset['text'])
    ner_scores = np.array(list(map(lambda o: np.mean([d['score'] for d in o]) if o else 0, outputs)))
    return ner_scores