import torch
from torch import nn
from transformers import BertModel, AutoModel


class BertWithNER(nn.Module):
    def __init__(self, bert_model, ner_model, n_labels=2):
        super(BertWithNER, self).__init__()
        self.base_model = BertModel.from_pretrained(bert_model)
        self.ner = BertModel.from_pretrained(ner_model)
        for param in self.ner.parameters():
            param.requires_grad = False 

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768*2, 768, bias=True),
            # nn.Tanh(),
            nn.LeakyReLU(), 
            nn.Dropout(p=0.1),
            nn.Linear(768, n_labels, bias=True)
        )

    def forward(self, input_ids, attention_mask, auxiliary_input_ids, **kwargs):
        logits_base = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits_ner = self.ner(auxiliary_input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        concatenated_vectors = torch.concat((logits_base, logits_ner), axis=1)
        output = self.classifier(concatenated_vectors)
        return {'logits':output}


class AutoModelWithNER(nn.Module):
    def __init__(self, model, ner_model, n_labels=2, single_layer_cls=False):
        super(AutoModelWithNER, self).__init__()
        self.base_model = AutoModel.from_pretrained(model)
        self.ner = BertModel.from_pretrained(ner_model)
        for param in self.ner.parameters():
            param.requires_grad = False 

        if single_layer_cls:
            self.classifier = nn.Linear(768*2, n_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768*2, 768, bias=True),
                # nn.Tanh(),
                nn.ReLU(), 
                nn.Dropout(p=0.1),
                nn.Linear(768, n_labels, bias=True)
            )
            

    def forward(self, input_ids, attention_mask, auxiliary_input_ids, **kwargs):
        logits_base = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits_ner = self.ner(auxiliary_input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        concatenated_vectors = torch.concat((logits_base, logits_ner), axis=1)
        output = self.classifier(concatenated_vectors)
        return {'logits':output}