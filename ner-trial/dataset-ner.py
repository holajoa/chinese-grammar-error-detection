import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import DatasetDict, Dataset
import numpy as np
from preprocess import *
from ner import *


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df, 
        model_name, 
        maxlength=128, 
        train_val_split=-1, 
        test=False, 
        eda=True, 
        ner_config=None, 
        ignore_keys=[], 
        **kwargs, 
    ):
        self.model_name = model_name
        self.maxlength = maxlength
        self.test_stage = test
        self.train_val_split = train_val_split
        self.ner_config = ner_config
        
        self.texts = self.eda(df, **kwargs) if eda else df.text
        if not self.test_stage:
            self.labels = df.label
        else:
            if 'labels' not in ignore_keys:
                ignore_keys.append('labels') 
        self.ignore_keys = ignore_keys


    def eda(self, df, **kwargs):
        cleaned = df.text.map(DataPreprocessor(**kwargs))
        return cleaned

    def ner(self, model, tokenizer):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ner_scores = get_ner_score(self.texts, model, tokenizer, device)
        return ner_scores

    def tokenize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.inputs = self.tokenizer(
            self.texts.tolist(), padding='max_length', max_length=self.maxlength,  
            truncation=True, return_tensors='pt', 
        )
        if self.ner_config:
            tokenizer = AutoTokenizer.from_pretrained(self.ner_config)
            model = AutoModelForTokenClassification.from_pretrained(self.ner_config)
            self.inputs['ner_scores'] = self.ner(model, tokenizer)

    def __getitem__(self, index):
        data_dict = {}
        for key, value in self.inputs.items():
            if key not in self.ignore_keys:
                indexed_value = torch.tensor(value[index]).squeeze()
                data_dict[key] = indexed_value
        if not self.test_stage:
            data_dict['labels'] = torch.tensor(self.labels[index].values).squeeze()
        dataset_obj = Dataset.from_dict(data_dict)
        return dataset_obj

        # input_ids = torch.tensor(self.inputs["input_ids"][index]).squeeze()
        # attn_ids = torch.tensor(self.inputs["attention_mask"][index]).squeeze()
        # if not self.test_stage:
        #     label_ids = torch.tensor(self.labels[index].values).squeeze()
        #     return Dataset.from_dict({"input_ids": input_ids, "labels": label_ids, "attention_mask":attn_ids,})
        # return Dataset.from_dict({"input_ids": input_ids, "attention_mask":attn_ids})

    def construct_dataset(self, val_idx=None):
        if val_idx is not None or 1 > self.train_val_split > 0:
            if val_idx is None: 
               val_idx = np.random.randint(0, self.__len__(), size=int(self.__len__() * (1-self.train_val_split)))
            train_idx = np.array(list(set(np.arange(self.__len__())) - set(val_idx)))
    
            self.dataset = DatasetDict(
                train=self.__getitem__(train_idx), 
                val=self.__getitem__(val_idx), 
            )
            self.dataset.set_format("torch")
        else:
            self.dataset = DatasetDict(train=self.__getitem__(np.arange(self.__len__())))
        self.dataset.set_format(output_all_columns=True)  # keep the ner_scores

    @classmethod
    def classes(self):
        return self.labels
    
    def __len__(self):
        return self.inputs['attention_mask'].size(0)
    