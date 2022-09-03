import numpy as np
import torch
from transformers import TrainingArguments, Trainer
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import postprocess_logits


class MyTrainingArguments(TrainingArguments):
    def __init__(self, epsilon, alpha=0.5, gamma=0, local_loss_param=1e-2, *args, **kwargs):
        super(MyTrainingArguments, self).__init__(*args, **kwargs)
        self.epsilon = epsilon    # add a perturbation parameter

        #### Define focal loss parameters: 
        # focal loss = -alpha*(1-pred)^{gamma} * log(pred)   if ture_label=1
        # focal loss = -(1-alpha)*pred^{gamma} * log(1-pred) if ture_label=0
        self.alpha = alpha
        self.gamma = gamma
        self.local_loss_param = local_loss_param


class AdversarialTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        
        # Extract embeddings
        # shape=(vocab_size, hidden_size)
        embeddings = model.base_model.embeddings.word_embeddings.weight

        # get gradients from embedding layer
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward(inputs=embeddings)
        grads = model.base_model.embeddings.word_embeddings.weight.grad.cpu()

        # Add perturbations (Fast Gradient Method/FGM)
        delta = self.args.epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)
        with torch.no_grad():
            model.base_model.embeddings.word_embeddings.weight += delta.cuda()

        # Compute loss and backprop as usual
        loss = self.compute_loss(model, inputs)


        ## ============= Copied from huggingface source code =============
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
    
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        ## ============= Copied from huggingface source code =============


        # Remove the perturbations
        with torch.no_grad():
            model.base_model.embeddings.word_embeddings.weight -= delta.cuda()

        return loss.detach()


    def log(self, logs):
        """Overwrite original log method to log to external file."""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        # log to external file
        logging.info(output)
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)

        true_labels = inputs['labels']
        if torch.is_tensor(outputs):
            logits = outputs
        else:
            logits = outputs['logits']
    
        loss = binary_focal_loss(logits, true_labels, alpha=self.args.alpha, gamma=self.args.gamma)

        ## Logging
        # logging.info(f"True labels:      {inputs['labels'].cpu().numpy()}")
        # logging.info(f'Predicted labels: {np.argmax(outputs.logits.cpu().detach().numpy(), axis=1)}')
        # logging.info(f'loss = {loss}\n')

        return (loss, outputs) if return_outputs else loss


class ImbalancedTrainer(Trainer):
    def log(self, logs):
        """Overwrite original log method to log to external file."""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        # log to external file
        logging.info(output)
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)

        true_labels = inputs['labels']
        # if self.model.token_level:
        #     logits = outputs['sequence_logits']
        try:
            logits = outputs['logits']
        except:
            logits = outputs['predictions']
        pass_pred_labels = 'predictions' in outputs.keys()

        attention_mask = inputs['attention_mask']
        focal_loss = binary_focal_loss(logits, true_labels, alpha=self.args.alpha, gamma=self.args.gamma, pass_pred_labels=pass_pred_labels)
        if 'sequence_logits' in outputs.keys():
            sequence_logits = outputs['sequence_logits']
            local_loss = token_label_loss(sequence_logits, true_labels, hyperparam=self.args.local_loss_param, alpha=self.args.alpha, attention_mask=attention_mask)
        else: 
            local_loss = 0
        loss = focal_loss + local_loss

        return (loss, outputs) if return_outputs else loss


def token_label_loss(sequence_logits, true_labels, attention_mask, sum=True, hyperparam=2e-3, alpha=1):
    from torch.distributions import Categorical

    pos_idx = torch.Tensor(true_labels == 1).long()
    neg_idx = torch.Tensor(true_labels == 0).long()

    logits_difference = sequence_logits[..., 1] - sequence_logits[..., 0]

    # For positive examples, maximise difference between max logit difference and min logit difference
    loss_pos = alpha * (logits_difference.min(-1).values - logits_difference.max(-1).values) / 2 * pos_idx   

    # For negative examples, minimise logit differences
    loss_neg = logits_difference.mean(-1) * neg_idx

    # entropies = Categorical(logits=sequence_logits).entropy()
    # # For positive examples, minimise entropy
    # loss_pos = alpha * entropies*attention_mask / attention_mask.sum(1).view(-1, 1) * pos_idx.view(-1, 1)

    # # For negative examples, maximise entropy
    # loss_neg = - entropies*attention_mask / (attention_mask.sum(1).view(-1, 1)) * neg_idx.view(-1, 1)

    if sum:
        loss_pos = torch.sum(loss_pos)
        loss_neg = torch.sum(loss_neg)

    loss = loss_pos + loss_neg
    
    return loss * hyperparam


def binary_focal_loss(logits, true_labels, alpha=1, gamma=0, sum=True, pass_pred_labels=False):
    def convert_labels_to_probs(pred_labels):
        sizea, sizeb = pred_labels.size()
        pred_probs = torch.zeros(size=(sizea, sizeb, 2), dtype=pred_labels.dtype)
        a, b = torch.argwhere(pred_labels).T
        pred_probs[a, b, 1] = 1
        a, b = torch.argwhere(pred_labels == 0).T
        pred_probs[a, b, 0] = 1
        return pred_probs

    pos_idx = torch.Tensor(true_labels == 1).long()
    neg_idx = torch.Tensor(true_labels == 0).long()

    if pass_pred_labels:
        pred_probs = torch.any(logits, dim=1).float()
        pred_probs.requires_grad = True
    else:
        pred_probs = torch.nn.Softmax(dim=-1)(logits)[..., -1]

    if pred_probs.ndim > 1:
        pos_idx = pos_idx.view(-1, 1)
        neg_idx = neg_idx.view(-1, 1)
    loss_pos = - alpha * ((1 - pred_probs)**gamma * torch.log(pred_probs+1e-7)) * pos_idx
    loss_neg = - (pred_probs**gamma * torch.log(1-pred_probs+1e-7)) * neg_idx

    if sum:
        loss_pos = torch.sum(loss_pos)
        loss_neg = torch.sum(loss_neg)

    loss = loss_pos + loss_neg
    
    return loss

