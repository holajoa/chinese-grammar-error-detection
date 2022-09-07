import torch
from torch import nn
from transformers import BertModel, AutoModel, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from layers import DynamicCRF
from utils import postprocess_logits
from torchcrf import CRF
from numpy import array
import synonyms

from typing import List


class AutoModelWithClassificationHead(nn.Module):
    def __init__(self, bert_model, n_labels=2, token_level=False, calibration_temperature=1., pooling_mode='max'):
        super(AutoModelWithClassificationHead, self).__init__()
        self.base_model = AutoModel.from_pretrained(bert_model) 
        self.calibration_temperature = calibration_temperature
        self.token_level = token_level
        self.pooling_mode = pooling_mode
        if self.pooling_mode not in ['cls', 'max', 'hybrid']:
            raise NotImplementedError("Choose pooling_model from ['cls', 'max', 'hybrid'].")
        self.classifier = nn.Linear(self.base_model.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        MLM_logits = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        sequence_logits = self.classifier(MLM_logits)
        if self.token_level:
            return {'sequence_logits':sequence_logits / self.calibration}
        if self.pooling_mode == 'cls':
            output = sequence_logits[:, 0, :]
        elif self.pooling_mode == 'max':
            output = postprocess_logits(sequence_logits, attention_mask, self.calibration_temperature)
        elif self.pooling_mode == 'hybrid':
            output = sequence_logits[:, 0, :] + postprocess_logits(sequence_logits, attention_mask, self.calibration_temperature)
        return {'logits':output, 'sequence_logits':sequence_logits}


class AutoModelWithOOBModel(nn.Module):
    def __init__(self, model, oob_model, n_labels=2, concatenate=True):
        super(AutoModelWithOOBModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model)
        self.oob_model = BertModel.from_pretrained(oob_model)
        for param in self.oob_model.parameters():
            param.requires_grad = False 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.base_model.config.hidden_size + self.oob_model.config.hidden_size, 128, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_labels, bias=True)
        )
        self.concatenate = concatenate
        # self.pooling_mode = pooling_mode


    def forward(self, input_ids, attention_mask, auxiliary_input_ids, **kwargs):
        logits_base = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits_oob = self.oob_model(auxiliary_input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        if self.concatenate:
            concatenated_vectors = torch.concat((logits_base, logits_oob), axis=1)
        else: 
            concatenated_vectors = logits_base + logits_oob
        output = self.classifier(concatenated_vectors)
        return {'logits':output}


class AutoModelBaseline(nn.Module):
    def __init__(self, model, hidden_layer_size=128, n_labels=2):
        super(AutoModelBaseline, self).__init__()
        self.base_model = AutoModel.from_pretrained(model)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.base_model.config.hidden_size, hidden_layer_size, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_layer_size, n_labels, bias=True)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        logits_base = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        output = self.classifier(logits_base)
        return {'logits':output}


class Similarity(nn.Module):
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, s1, s2):
        return synonyms.compare(s1, s2)


class MLMWithSimilarity(nn.Module):
    def __init__(self, model='hfl/chinese-macbert-base', input_seq_len=128, hidden_layer_size=128, n_labels=2):
        super(MLMWithSimilarity, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(model)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_seq_len, hidden_layer_size, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_layer_size, n_labels, bias=True)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        logits = self.base_model(input_ids, attention_mask=attention_mask).logits    # shape (batch, seq_len, vocab_size)
        fill_word_ids = logits.argmax(-1)
        org_emb = self.base_model.bert.embeddings.word_embeddings(input=input_ids)
        pred_emb = self.base_model.bert.embeddings.word_embeddings(input=fill_word_ids)
        sentence_similarity = torch.nn.CosineSimilarity(dim=2)(org_emb, pred_emb)
        logits = self.cls(sentence_similarity)
        return {'logits':logits, 'bert_logits':logits, 'output_token_id':fill_word_ids}


class DualModelForGEC(nn.Module):
    def __init__(self, model_name_1, model_name_2, n_labels=2, hidden_layer_size=128, *args, **kwargs):
        super(DualModelForGEC, self).__init__()
        self.model_1 = AutoModel.from_pretrained(model_name_1)
        self.model_2 = AutoModel.from_pretrained(model_name_2)

        concat_logits_size = self.model_1.config.hidden_size + self.model_2.config.hidden_size

        self.cls = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(concat_logits_size, hidden_layer_size, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_layer_size, n_labels, bias=True)
        )

    def forward(
            self,
            input_ids,
            attention_mask_1,
            attention_mask_2,
            maxlength, 
            **kwargs, 
    ):
        logits_1 = self.model_1(input_ids=input_ids[:, :maxlength], attention_mask=attention_mask_1)[:, 0, :]
        logits_2 = self.model_2(input_ids=input_ids[:, maxlength:], attention_mask=attention_mask_2)[:, 0, :]
        concat_logits = torch.concat((logits_1, logits_2), 1)

        output_logits = self.cls(concat_logits)

        return {'logits':output_logits, 'model_1_logits':logits_1, 'model_2_logits':logits_2}