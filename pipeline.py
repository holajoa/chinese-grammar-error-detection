from transformers import AutoModelForSequenceClassification
from typing import List, Union

from utils import *
from dataset import *
from torch.utils.data import DataLoader


class PipelineGED:
    def __init__(self, model_name:str, data_configs=None):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, 
        )
        if data_configs:
            self.data_configs = data_configs
        else:
            self.data_configs = {
                'model_name':model_name,
                'maxlength':128,
                'train_val_split':-1,
                'test':True, 
                'remove_username':False,
                'remove_punctuation':False, 
                'to_simplified':False, 
                'emoji_to_text':False, 
                'split_words':False, 
                'cut_all':False, 
        }

    def feedforward(
        self, 
        ds:DatasetWithAuxiliaryEmbeddings, 
        checkpoints:List[str], 
        device:torch.device, 
        raw_outputs:bool=True, 
        output_probabilities:bool=False, 
    ) -> np.ndarray:
        output_tensors = []

        for cp in checkpoints:
            state_dict = torch.load(cp, map_location=device)
            # for key in list(state_dict.keys()):
            #     state_dict[key.replace('bert', 'base_model')] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)
            if 'cuda' in device.type:
                self.model.cuda()

            logits = []
            dataloader = DataLoader(ds.dataset['train'].with_format('torch'), batch_size=16)

            for batch in dataloader:
                inputs = {k:v.to(device) for k,v in batch.items()
                        if k in ds.tokenizer.model_input_names or k == 'auxiliary_input_ids'}
                with torch.no_grad():
                    output = self.model(**inputs)
                logits.append(output['logits'])

            output_tensors.append(torch.concat(logits))

        if output_probabilities:
            from torch.nn import Softmax
            sm = Softmax(dim=1)
            return sm(torch.stack(output_tensors).mean(0)).cpu().numpy()
        if raw_outputs:
            return torch.stack(output_tensors).mean(0).cpu().numpy()
        return torch.stack(output_tensors).mean(0).argmax(1).cpu().numpy()

    @staticmethod
    def map_to_df(texts:str) -> pd.DataFrame:
        if isinstance(texts, str):
            texts = [texts]
        return pd.DataFrame(data=texts, columns=['text'])

    def __call__(
        self, 
        texts:Union[List[str], str], 
        checkpoints:List[str], 
        device:torch.device, 
        raw_outputs:bool=True, 
        output_probabilities:bool=False, 
    ) -> np.ndarray:
        test = DatasetWithAuxiliaryEmbeddings(df=self.map_to_df(texts), **self.data_configs)
        test.prepare_dataset()
        for txt in test.texts.values:
            print(txt)
        return self.feedforward(test, checkpoints, device, raw_outputs, output_probabilities)