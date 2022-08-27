from transformers import AutoModelForSequenceClassification
from typing import List, Union

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models import BertWithClassificationHead


class PipelineGED:
    def __init__(self, model_name:str, data_configs=None, single_layer_cls=True):
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name, num_labels=2, 
        # )
        self.model = BertWithClassificationHead(
            model_name, 
            n_labels=2, 
            single_layer_cls=single_layer_cls, 
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
        majority_vote=False, 
    ) -> np.ndarray:
        output_tensors = []
        output_sequence_logits = []

        for cp in checkpoints:
            state_dict = torch.load(cp, map_location=device)
            # for key in list(state_dict.keys()):
            #     state_dict[key.replace('bert', 'base_model')] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)
            if 'cuda' in device.type:
                self.model.cuda()

            logits = []
            sequence_logits = []
            dataloader = DataLoader(ds.dataset['train'].with_format('torch'), batch_size=16)

            for batch in dataloader:
                inputs = {k:v.to(device) for k,v in batch.items()
                        if k in ds.tokenizer.model_input_names or k == 'auxiliary_input_ids'}
                with torch.no_grad():
                    output = self.model(**inputs)
                logits.append(output['logits'])
                sequence_logits.append(output['sequence_logits'])
            output_tensors.append(torch.concat(logits))
            output_sequence_logits.append(torch.concat(sequence_logits, dim=0))

        output_sequence_logits_agg = torch.stack(output_sequence_logits, dim=3)
        if majority_vote:
            return voting(torch.stack(output_tensors))
        else:
            if output_probabilities:
                from torch.nn import Softmax
                
                return (
                    Softmax(dim=1)(torch.stack(output_tensors).mean(0)).cpu().numpy(), 
                    Softmax(dim=2)(output_sequence_logits_agg.mean(-1)).cpu().numpy(), 
                )
                
            if raw_outputs:
                return (
                    torch.stack(output_tensors).mean(0).cpu().numpy(), 
                    output_sequence_logits_agg.mean(-1).cpu().numpy()
                )
            return (torch.stack(output_tensors).mean(0).argmax(1).cpu().numpy(), output_sequence_logits_agg.mean(-1).argmax(2).cpu().numpy())

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
        display=True, 
        majority_vote=False, 
    ) -> np.ndarray:
        test = DatasetWithAuxiliaryEmbeddings(df=self.map_to_df(texts), **self.data_configs)
        test.prepare_dataset()
        # for txt in test.texts.values:
        #     print(txt)
        if majority_vote:
            return self.feedforward(test, checkpoints, device, raw_outputs, output_probabilities, majority_vote=True)
        probs, seq_probs = self.feedforward(test, checkpoints, device, raw_outputs, output_probabilities)
        err_char_lst = self.display_error_chars(seq_probs, test.texts.values, display=display)
        return probs, seq_probs, err_char_lst

    @staticmethod
    def display_error_chars(seq_probs, texts, display=True):
        err_char_lst = []
        for probs, txt in zip(seq_probs, texts):
            err_idx = np.argwhere(probs[:(2+len(txt)), 1] > probs[:(2+len(txt)), 0])
            err_chars = np.array(['[CLS]'] + list(txt) + ['[SEP]'])[err_idx].flatten()
            if display:
                print(txt)
                print(err_chars)
            err_char_lst.append(err_chars)
        return err_char_lst
            
            
