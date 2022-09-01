import torch
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
import numpy as np
from preprocess import *
from typing import Dict, List, Union


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df, 
        model_name, 
        maxlength=128, 
        train_val_split=-1, 
        test=False, 
        eda=True, 
        device='cpu', 
        **kwargs, 
    ):
        self.model_name = model_name
        self.maxlength = maxlength
        self.test_stage = test
        self.train_val_split = train_val_split
        self.texts = self.eda(df, **kwargs) if eda else df.text
        if not self.test_stage:
            self.labels = df.label
        self.device = device
    
    def prepare_dataset(self, val_idx=None):
        self.val_idx = val_idx
        self.tokenize()
        self.construct_dataset(val_idx)

    def eda(self, df, **kwargs):
        cleaned = df.text.map(DataPreprocessor(**kwargs))
        return cleaned

    def tokenize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.inputs = self.tokenizer(
                    self.texts.tolist(), padding='max_length', max_length=self.maxlength,  
                    truncation=True, return_tensors='pt'
                ).to(self.device)

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
        return len(self.texts)
    

class DatasetWithAuxiliaryEmbeddings(torch.utils.data.Dataset):
    def __init__(
        self, 
        df:pd.DataFrame, 
        model_name:str, 
        aux_model_name:str=None, 
        maxlength:int=128, 
        train_val_split:Union[float, int]=-1, 
        test:bool=False, 
        eda:bool=True, 
        device:str='cpu',
        da_configs:Dict[str, dict]=None, 
        **kwargs
    ):
        self.model_name = model_name
        self.aux_model_name = aux_model_name
        self.maxlength = maxlength
        self.test_stage = test
        self.train_val_split = train_val_split

        df_aug = DataAugmentation(da_configs).aug(df) if da_configs else df
        self.texts = self.eda(df_aug, **kwargs) if eda else df_aug.text
        self.device = device
        if not self.test_stage:
            self.labels = df.label
        self.dataset = None

    def prepare_dataset(self, val_idx=None):
        self.val_idx = val_idx
        self.tokenize()
        self.construct_dataset(val_idx)
    
    def tokenize(self):
        self.initialise_tokenizer()
        self.inputs = self.tokenizer(
            self.texts.tolist(), padding='max_length', max_length=self.maxlength,  
            truncation=True, return_tensors='pt', 
        ).to(self.device)
        if self.aux_model_name:
            self.inputs['auxiliary_input_ids'] = self.aux_tokenizer(
                self.texts.tolist(), padding='max_length', max_length=self.maxlength,  
                truncation=True, return_tensors='pt', 
            )['input_ids'].to(self.device)

    def eda(self, df, **kwargs):
        cleaned = df.text.map(DataPreprocessor(**kwargs))
        return cleaned

    def initialise_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.aux_model_name:
            self.aux_tokenizer = AutoTokenizer.from_pretrained(self.aux_model_name)

    def __getitem__(self, index):
        data_dict = {}
        items_iterator = self.inputs.items() if self.dataset is None else self.dataset['train'].to_dict().items()
        for key, value in items_iterator:
            if isinstance(value, list):
                value = torch.from_numpy(np.array(value))
            indexed_value = value[index].to(device=self.device)
            data_dict[key] = indexed_value
        if not self.test_stage:
            data_dict['labels'] = torch.from_numpy(self.labels.iloc[index].values).to(device=self.device)
        dataset_obj = Dataset.from_dict(data_dict)
        return dataset_obj

    def construct_dataset(self, val_idx=None):
        if val_idx is not None or 1 > self.train_val_split > 0:
            if val_idx is None: 
               val_idx = np.random.randint(0, self.__len__(), size=int(self.__len__() * (1-self.train_val_split)))
            self.val_idx = val_idx
            self.train_val_split = len(self.val_idx) / self.__len__()
            train_idx = np.array(list(set(np.arange(self.__len__())) - set(val_idx)))
    
            self.dataset = DatasetDict(
                train=self.__getitem__(train_idx), 
                val=self.__getitem__(val_idx), 
            )
        else:
            self.dataset = DatasetDict(train=self.__getitem__(np.arange(self.__len__())))
        self.dataset.set_format('torch')

    @classmethod
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.texts)