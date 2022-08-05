import torch
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
import numpy as np
from preprocess import *


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, df, model_name, maxlength=128, train_val_split=-1, test=False, eda=True, **kwargs):
        self.model_name = model_name
        self.maxlength = maxlength
        self.test_stage = test
        self.train_val_split = train_val_split
        self.texts = self.eda(df, **kwargs) if eda else df.text
        if not self.test_stage:
            self.labels = df.label

    def eda(self, df, **kwargs):
        cleaned = df.text.map(DataPreprocessor(**kwargs))
        return cleaned

    def tokenize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.inputs = self.tokenizer(
                    self.texts.tolist(), padding='max_length', max_length=128,  
                    truncation=True, return_tensors='pt', 
                )

    def __getitem__(self, index):
        data_dict = {}
        for key, value in self.inputs.items():
            indexed_value = torch.tensor(value[index]).squeeze()
            data_dict[key] = indexed_value
        if not self.test_stage:
            data_dict['labels'] = torch.tensor(self.labels[index].values).squeeze()
        dataset_obj = Dataset.from_dict(data_dict)
        return dataset_obj

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

    @classmethod
    def classes(self):
        return self.labels
    
    def __len__(self):
        return self.inputs['attention_mask'].size(0)
    