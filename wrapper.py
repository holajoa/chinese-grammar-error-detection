import numpy as np
import torch
from transformers import TrainingArguments, Trainer


class AdversarialTrainingArguments(TrainingArguments):
    def __init__(self, epsilon, *args, **kwargs):
        super(AdversarialTrainingArguments, self).__init__(*args, **kwargs)
        self.epsilon = epsilon    # add a perturbation parameter


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

