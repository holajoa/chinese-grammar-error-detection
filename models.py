import torch
from torch import nn
from transformers import BertModel, AutoModel, AutoModelForMaskedLM
from utils import postprocess_logits

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
    def __init__(self, bert_model, n_labels=2, cls_hidden_size=768, single_layer_cls=False):
        super(BertWithClassificationHead, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(bert_model) 

        if single_layer_cls:
            # self.classifier = nn.Linear(self.base_model.config.vocab_size, n_labels)
            self.classifier = nn.Linear(self.base_model.config.hidden_size, n_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.base_model.config.hidden_size, cls_hidden_size, bias=True),
                # nn.Linear(self.base_model.config.vocab_size, cls_hidden_size, bias=True),
                # nn.Tanh(),
                nn.ReLU(), 
                nn.Dropout(p=0.1),
                nn.Linear(cls_hidden_size, n_labels, bias=True)
            )

    def forward(self, input_ids, attention_mask, **kwargs):
        MLM_logits = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        output = postprocess_logits(self.classifier(MLM_logits))
        return {'logits':output}
