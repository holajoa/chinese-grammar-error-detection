from transformers import AutoModelForTokenClassification
from typing import List, Union

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models import BertWithCRFHead, AutoModelWithClassificationHead


def map_to_df(texts:str) -> pd.DataFrame:
    if isinstance(texts, str):
        texts = [texts]
    return pd.DataFrame(data=texts, columns=['text'])


class POSTaggingPipeline:
    def __init__(self, model_name:str, data_configs=None):
        self.model = self._init_model(model_name)
        self.data_configs = data_configs if data_configs else self._get_default_data_configs(model_name)
        
    def _init_model(self, model_name):
        return AutoModelForTokenClassification.from_pretrained(model_name)

    def _get_default_data_configs(self, model_name):
        return {
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
        device:torch.device, 
    ):  
        from tqdm import tqdm

        if 'cuda' in device.type:
            self.model.cuda()

        logits = []
        dataloader = DataLoader(ds.dataset['train'].with_format('torch'), batch_size=16)

        for batch in tqdm(dataloader):
            inputs = {k:v.to(device) for k,v in batch.items()
                    if k in ds.tokenizer.model_input_names or k == 'auxiliary_input_ids'}
            with torch.no_grad():
                output = self.model(**inputs)

            logits.append(output['logits'])
        return torch.concat(logits)

    def __call__(
        self, 
        ds:DatasetWithAuxiliaryEmbeddings=None, 
        texts:Union[List[str], str]=None, 
        device:torch.device=torch.device('cpu'), 
        return_tags=True, 
    ):  
        if ds is not None:
            if not isinstance(ds, DatasetWithAuxiliaryEmbeddings):
                raise TypeError('Passed data object is not a Dataset. Pass it as argument `texts` instead.')
        else:
            ds = DatasetWithAuxiliaryEmbeddings(df=map_to_df(texts), **self.data_configs)
            ds.prepare_dataset()
        logits = self.feedforward(ds, device)
        tag_ids = torch.argmax(logits, dim=2)
        if return_tags:
            return tag_ids
        ds.dataset['train'] = ds.dataset['train'].add_column(name='tag_ids', column=[id for id in tag_ids.cpu().numpy()])
        ds.dataset['train'] = ds.dataset['train'].add_column(name='emissions', column=logits.cpu().numpy().tolist())
        return ds
            

class PipelineGED:
    def __init__(self, model_name:str, model_architecture:str='bert_with_clf_head', data_configs=None, pooling_mode='cls'):
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name, num_labels=2, 
        # )
        if model_architecture == 'bert_with_clf_head':
            self.model = AutoModelWithClassificationHead(
                model_name, 
                n_labels=2, 
                pooling_mode=pooling_mode, 
            )
        elif model_architecture == 'bert_with_crf_head':
            self.model = BertWithCRFHead(
                model_name, 
                n_labels=2, 
            )
        else:
            print(f'Model architecture {model_architecture} is not implemented.')
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
        from tqdm import tqdm

        for cp in checkpoints:
            state_dict = torch.load(cp, map_location=device)
            # for key in list(state_dict.keys()):
            #     state_dict[key.replace('bert', 'base_model')] = state_dict.pop(key)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if bool(missing_keys) | bool(unexpected_keys):
                print(f'Warning: state_dict does not match perfectly. \nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}')
            if 'cuda' in device.type:
                self.model.cuda()

            logits = []
            sequence_logits = []
            dataloader = DataLoader(ds.dataset['train'].with_format('torch'), batch_size=16)

            for batch in tqdm(dataloader):
                inputs = {k:v.to(device) for k,v in batch.items()
                        if k in ds.tokenizer.model_input_names or k == 'auxiliary_input_ids'}
                with torch.no_grad():
                    output = self.model(**inputs)
                try:
                    logits.append(output['logits'])    # not used
                    sequence_logits.append(output['sequence_logits'])
                    pass_logits = True
                except:
                    logits.append(output['predictions'])
                    pass_logits = False
            output_tensors.append(torch.concat(logits))
            if pass_logits:
                output_sequence_logits.append(torch.concat(sequence_logits, dim=0))
        if pass_logits:
            output_sequence_logits_agg = torch.stack(output_sequence_logits, dim=3).mean(-1)
            if self.model.pooling_mode == 'max':
                output_logits_agg = postprocess_logits(output_sequence_logits_agg, ds.dataset['train']['attention_mask'])
            elif self.model.pooling_mode == 'cls':
                output_logits_agg = output_sequence_logits_agg[:, 0, :]
            elif self.model.pooling_mode == 'hybrid':
                output = output_sequence_logits_agg[:, 0, :] + \
                    postprocess_logits(output_sequence_logits_agg, ds.dataset['train']['attention_mask'])
            else:
                raise NotImplementedError(f'pooling mode {self.model.pooling_mode} is not implemented.')
            if majority_vote:
                return voting(torch.stack(output_tensors))
            else:
                if output_probabilities:
                    from torch.nn import Softmax
                    
                    return (
                        Softmax(dim=1)(output_logits_agg).cpu().numpy(), 
                        Softmax(dim=2)(output_sequence_logits_agg).cpu().numpy(), 
                    )
                    
                if raw_outputs:
                    return (
                        output_logits_agg.cpu().numpy(), 
                        output_sequence_logits_agg.mean(-1).cpu().numpy()
                    )
                return (output_logits_agg.argmax(1).cpu().numpy(), output_sequence_logits_agg.mean(-1).argmax(2).cpu().numpy())
        else:
            assert majority_vote, 'Must use majority voting if model outputs labels directly.'
            pred_labels = torch.stack(output_tensors) # has shape (n_models, n_examples, seq_len). 
            n_models, n_samples, seq_len = pred_labels.size()
            pred_labels_flattened = pred_labels.view((n_models, n_samples*seq_len))
            agg_labels = []
            for single_examples_pred_labels in pred_labels_flattened.T:
                labels, counts = torch.unique(single_examples_pred_labels, return_counts=True)
                agg_labels.append(labels[torch.argmax(counts)])
            seq_predictions = torch.tensor(agg_labels).view(n_samples, seq_len)
            predictions = torch.any(seq_predictions, dim=1).int()
            return predictions, seq_predictions

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
        test = DatasetWithAuxiliaryEmbeddings(df=map_to_df(texts), **self.data_configs)
        test.prepare_dataset()
        if majority_vote:
            return self.feedforward(test, checkpoints, device, raw_outputs, output_probabilities, majority_vote=True)
        probs, seq_probs = self.feedforward(test, checkpoints, device, raw_outputs, output_probabilities)
        err_char_lst = self.display_error_chars(seq_probs, test, display=display)
        return probs, seq_probs, err_char_lst

    @staticmethod
    def display_error_chars(seq_probs, test:DatasetWithAuxiliaryEmbeddings, display=True):
        import matplotlib.pyplot as plt

        err_char_lst = []
        for probs, txt_ids, mask in zip(seq_probs, test.inputs['input_ids'], test.inputs['attention_mask']):
            txt_ids = txt_ids.masked_select(mask.bool())
            err_idx = np.argwhere(probs[:txt_ids.size(-1), 1] > probs[:txt_ids.size(-1), 0]).flatten()
            err_chars = test.tokenizer.convert_ids_to_tokens(txt_ids[err_idx])
            if display:
                print(' '.join(test.tokenizer.convert_ids_to_tokens(txt_ids)))
                print(err_chars)
            err_char_lst.append(err_chars)
        if display:
            masks = test.inputs['attention_mask'].cpu().numpy()
            max_len = masks.sum(1).max()
            seq_probs_masked = seq_probs[..., 1] * masks
            plt.figure(figsize=(20, 1))
            plt.imshow(seq_probs_masked[:, :max_len], cmap='binary')
        return err_char_lst
            