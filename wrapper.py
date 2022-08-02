import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss


class AdversarialTrainingArguments(TrainingArguments):
    def __init__(self, epsilon, alpha=0.5, gamma=0, *args, **kwargs):
        super(AdversarialTrainingArguments, self).__init__(*args, **kwargs)
        self.epsilon = epsilon    # add a perturbation parameter

        #### Define focal loss parameters: 
        # focal loss = -alpha*(1-pred)^{gamma} * log(pred)   if ture_label=1
        # focal loss = -(1-alpha)*pred^{gamma} * log(1-pred) if ture_label=0
        self.alpha = alpha
        self.gamma = gamma    


class AdversarialTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        
        # Extract embeddings
        # shape=(vocab_size, hidden_size)
        embeddings = model.base_model.embeddings.word_embeddings.weight

        # get gradients from embeddinglayer
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
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        true_labels = inputs['labels']
        logits = outputs.logits
        # pred_labels = torch.argmax(logits, axis=1)
        loss = binary_focal_loss(logits, true_labels, alpha=self.args.alpha, gamma=self.args.gamma)

        ## Logging
        # logging.info(f"True labels:      {inputs['labels'].cpu().numpy()}")
        # logging.info(f'Predicted labels: {np.argmax(outputs.logits.cpu().detach().numpy(), axis=1)}')
        # logging.info(f'loss = {loss}\n')

        return (loss, outputs) if return_outputs else loss

def binary_focal_loss(logits, true_labels, alpha=0.5, gamma=0):
    pos_idx = torch.Tensor(true_labels == 1).long()
    neg_idx = torch.Tensor(true_labels == 0).long()
    pred_probs = torch.nn.Softmax(dim=1)(logits)[:, 1]

    loss_pos = - alpha * torch.sum(((1 - pred_probs)**gamma * torch.log(pred_probs)) * pos_idx)
    loss_neg = - (1-alpha) * torch.sum((pred_probs**gamma * torch.log(1-pred_probs)) * neg_idx)
    loss = loss_pos + loss_neg
    return loss
