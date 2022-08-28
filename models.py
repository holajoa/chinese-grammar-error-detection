import torch
from torch import nn
from transformers import BertModel, AutoModel, AutoModelForMaskedLM
from utils import postprocess_logits
from torchcrf import CRF
from numpy import array

from typing import List


class BertWithNER(nn.Module):
    def __init__(self, bert_model, ner_model, n_labels=2, concatenate=True):
        super(BertWithNER, self).__init__()
        self.base_model = BertModel.from_pretrained(bert_model)
        self.ner = BertModel.from_pretrained(ner_model)
        self.concatenate = concatenate
        for param in self.ner.parameters():
            param.requires_grad = False 

        if self.concatenate:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768*2, 768, bias=True),
                # nn.Tanh(),
                nn.ReLU(), 
                nn.Dropout(p=0.1),
                nn.Linear(768, n_labels, bias=True)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768, 768, bias=True),
                # nn.Tanh(),
                nn.ReLU(), 
                nn.Dropout(p=0.1),
                nn.Linear(768, n_labels, bias=True)
            )

    def forward(self, input_ids, attention_mask, auxiliary_input_ids, **kwargs):
        logits_base = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits_ner = self.ner(auxiliary_input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        if self.concatenate:
            concatenated_vectors = torch.concat((logits_base, logits_ner), axis=1)
        else: 
            concatenated_vectors = logits_base + logits_ner
        output = self.classifier(concatenated_vectors)
        return {'logits':output}


class AutoModelWithNER(nn.Module):
    def __init__(self, model, ner_model, n_labels=2, single_layer_cls=False, concatenate=True):
        super(AutoModelWithNER, self).__init__()
        self.base_model = AutoModel.from_pretrained(model)
        self.ner = BertModel.from_pretrained(ner_model)
        self.concatenate = concatenate
        for param in self.ner.parameters():
            param.requires_grad = False 

        if single_layer_cls:
            self.classifier = nn.Linear(768*2, n_labels) if self.concatenate else nn.Linear(768, n_labels)
        else:
            if self.concatenate:
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(768*2, 768, bias=True),
                    # nn.Tanh(),
                    nn.ReLU(), 
                    nn.Dropout(p=0.1),
                    nn.Linear(768, n_labels, bias=True)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(768, 768, bias=True),
                    # nn.Tanh(),
                    nn.ReLU(), 
                    nn.Dropout(p=0.1),
                    nn.Linear(768, n_labels, bias=True)
                )

    def forward(self, input_ids, attention_mask, auxiliary_input_ids, **kwargs):
        logits_base = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits_ner = self.ner(auxiliary_input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        if self.concatenate:
            concatenated_vectors = torch.concat((logits_base, logits_ner), axis=1)
        else: 
            concatenated_vectors = logits_base + logits_ner
        output = self.classifier(concatenated_vectors)
        return {'logits':output}

class BertWithClassificationHead(nn.Module):
    def __init__(self, bert_model, n_labels=2, cls_hidden_size=768, single_layer_cls=False, calibration_temperature=1.):
        super(BertWithClassificationHead, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(bert_model) 
        self.calibration_temperature = calibration_temperature

        if single_layer_cls:
            self.classifier = nn.Linear(self.base_model.config.hidden_size, n_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.base_model.config.hidden_size, cls_hidden_size, bias=True),
                # nn.Tanh(),
                nn.ReLU(), 
                nn.Dropout(p=0.1),
                nn.Linear(cls_hidden_size, n_labels, bias=True)
            )

    def forward(self, input_ids, attention_mask, **kwargs):
        MLM_logits = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        sequence_logits = self.classifier(MLM_logits)
        output = postprocess_logits(sequence_logits, attention_mask, self.calibration_temperature)
        return {'logits':output, 'sequence_logits':sequence_logits}


class BertWithCRFHead(nn.Module):
    def __init__(self, bert_model, n_labels=2, cls_hidden_size=768, single_layer_cls=False, calibration_temperature=1.):
        super(BertWithCRFHead, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(bert_model) 
        self.calibration_temperature = calibration_temperature

        if single_layer_cls:
            self.classifier = nn.Linear(self.base_model.config.hidden_size, n_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.base_model.config.hidden_size, cls_hidden_size, bias=True),
                # nn.Tanh(),
                nn.ReLU(), 
                nn.Dropout(p=0.1),
                nn.Linear(cls_hidden_size, n_labels, bias=True)
            )
        self.crf = CRF(n_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, **kwargs):
        MLM_logits = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        sequence_logits = self.classifier(MLM_logits)
        sequence_tags_raw = self.crf.decode(sequence_logits, attention_mask)
        sequence_tags = self._pad_sequence_tags(sequence_tags_raw, input_ids)
        return {'predictions':sequence_tags}

    @staticmethod
    def _pad_sequence_tags(sequence_tags_raw:List[list], input_ids:torch.Tensor):
        sequence_tags = torch.zeros(size=input_ids.size(), dtype=input_ids.dtype, device=input_ids.device)
        for i, tags in enumerate(sequence_tags_raw):
            sequence_tags[i][:len(tags)] = torch.from_numpy(array(tags))
        return sequence_tags
